#!/usr/bin/env python3
"""Stream a scene activation shard into the compact ck8 phase-input format.

The input ``X`` member is never materialised as a whole.  It is read from the
NPZ zip stream in record batches, pooled over the ``all`` token segment at
physical layer 12 and denoise index 3, and assigned with the production
``ClusterPhaseAssigner``.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.failure_online.cluster_phase import ClusterPhaseAssigner


def _scalar(v):
    a = np.asarray(v)
    if a.shape == ():
        return a.item()
    if a.size == 1:
        return a.reshape(-1)[0].item()
    return a


def _json_value(v):
    v = _scalar(v)
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return v


def _read_meta(z):
    if "meta_json" not in z.files:
        raise ValueError("shard: meta_json is required")
    raw = _json_value(z["meta_json"])
    if isinstance(raw, str):
        return json.loads(raw)
    if isinstance(raw, dict):
        return raw
    raise ValueError("shard: meta_json is not JSON text")


def _header(stream):
    version = np.lib.format.read_magic(stream)
    readers = {(1, 0): np.lib.format.read_array_header_1_0,
               (2, 0): np.lib.format.read_array_header_2_0,
               (3, 0): np.lib.format.read_array_header_2_0}
    if version not in readers:
        raise ValueError(f"unsupported X.npy format {version}")
    shape, fortran, dtype = readers[version](stream)
    if fortran:
        raise ValueError("X.npy is Fortran ordered; row streaming is refused")
    if len(shape) != 5 or shape[1] < 1 or shape[2] != 4 or shape[3] < 1 or shape[4] != 1536:
        raise ValueError(f"X shape {shape}; expected [N,L,4,T,1536]")
    return tuple(int(x) for x in shape), np.dtype(dtype)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _field(z, name, n):
    if name not in z.files:
        raise ValueError(f"shard: required field {name!r} is missing")
    a = np.asarray(z[name])
    if len(a) != n:
        raise ValueError(f"shard: {name} length {len(a)} != X rows {n}")
    return a.copy()


def _validate(fields, meta):
    n = len(fields["ep_id"])
    keys = ["ep_id", "scene", "noise", "jitter", "succ", "ep_len"]
    rows = {}
    for i in range(n):
        ep = int(fields["ep_id"][i])
        key = (ep,)
        rows.setdefault(key, []).append(i)
    for (ep,), inds in rows.items():
        rec = np.asarray(fields["rec_idx"])[inds].astype(np.int64)
        if not np.array_equal(rec, np.arange(len(rec), dtype=np.int64)):
            raise ValueError(f"ep_id={ep}: rec_idx is not ordered 0..N-1")
        for name in keys[1:]:
            vals = [str(_scalar(fields[name][i])) for i in inds]
            if len(set(vals)) != 1:
                raise ValueError(f"ep_id={ep}: {name} is not episode-constant")
    ep_order = np.asarray(fields["ep_id"]).astype(np.int64)
    if len(ep_order) and np.any(ep_order[1:] < ep_order[:-1]):
        raise ValueError("ep_id rows are not sorted")
    if len(ep_order):
        starts = np.flatnonzero(np.r_[True, ep_order[1:] != ep_order[:-1]])
        if len(set(ep_order[starts])) != len(starts):
            raise ValueError("ep_id records are interleaved")
    configs = {}
    for i in range(n):
        cfg = tuple(str(_scalar(fields[k][i])) for k in ("scene", "noise", "jitter"))
        ep = int(fields["ep_id"][i])
        prior = configs.setdefault(cfg, ep)
        if prior != ep:
            raise ValueError(f"duplicate scene/noise/jitter configuration {cfg} for ep_id={prior},{ep}")
    if not meta.get("capture_layers"):
        raise ValueError("meta_json.capture_layers is missing")


def prepare(shard: Path, bundle: Path, slug: str, out: Path, batch: int = 64):
    if batch < 1: raise ValueError("batch must be positive")
    if out.exists():
        raise FileExistsError(f"refusing to reuse existing output: {out}")
    assigner = ClusterPhaseAssigner.from_bundle(bundle, task=slug, device="cpu")
    with np.load(shard, allow_pickle=False) as z:
        meta = _read_meta(z)
        layers = [int(x) for x in meta["capture_layers"]]
        li = assigner.resolve_layer_index(layers)
        if int(assigner.feature.get("layer", 12)) != 12 or int(assigner.feature.get("denoise_index", 3)) != 3:
            raise ValueError(f"bundle feature must be layer 12/denoise 3: {assigner.feature}")
        if str(assigner.feature.get("seg", "all")) != "all":
            raise ValueError(f"bundle feature must use segment all: {assigner.feature}")
        if assigner.centers.shape[0] != 8:
            raise ValueError(f"ck8 bundle must have 8 centers, got {assigner.centers.shape[0]}")
        si = list(meta["segment_names"]).index("all")
        # Accessing fields is small; accessing X is deliberately deferred to the zip stream.
        with zipfile.ZipFile(shard) as zin:
            if "X.npy" not in zin.namelist():
                raise ValueError("shard: X.npy member is missing")
            if "phase_code" not in z.files:
                raise ValueError("shard: phase_code is required as original GT label")
            with zin.open("X.npy") as xs:
                shape, dtype = _header(xs)
                n, _, _, tokens, dim = shape
                if tokens != len(meta["segment_names"]):
                    raise ValueError("segment axis does not match metadata")
                fields = {k: _field(z, k, n) for k in
                          ("ep_id", "scene", "noise", "jitter", "rec_idx", "succ", "ep_len")}
                for optional in ("jitter_idx", "jitter_reset_idx"):
                    if optional in z.files:
                        fields[optional] = _field(z, optional, n)
                _validate(fields, meta)
                xout = np.zeros((n, 1, 4, 1, 1536), dtype=np.float32)
                phases = np.empty(n, dtype=np.int16)
                item = int(np.prod(shape[1:]) * dtype.itemsize)
                for start in range(0, n, batch):
                    count = min(batch, n - start)
                    raw = xs.read(count * item)
                    if len(raw) != count * item:
                        raise ValueError("truncated X.npy member")
                    arr = np.frombuffer(raw, dtype=dtype).reshape((count,) + shape[1:])
                    if not np.issubdtype(arr.dtype, np.floating):
                        raise ValueError(f"X dtype {arr.dtype} is not floating point")
                    feat = arr[:, li, 3, si, :].astype(np.float32)
                    if not np.isfinite(feat).all():
                        raise ValueError(f"non-finite selected feature at rows {start}:{start + count}")
                    xout[start:start + count, 0, 3, 0, :] = feat
                    phases[start:start + count] = [o["idx"] for o in assigner.assign_batch(feat)]
        succ = fields["succ"]
        # phase_code is the original GT phase label; it is retained separately.
        with np.load(shard, allow_pickle=False) as z2:
            gt_phase = np.asarray(z2["phase_code"]).copy()
            if len(gt_phase) != n:
                raise ValueError(f"shard: phase_code length {len(gt_phase)} != X rows {n}")
        _validate(fields, meta)
        out_meta = dict(meta)
        out_meta.update({"capture_layers": [12], "segment_names": ["all"],
                         "phase_codebook": {f"c{i}": i for i in range(8)},
                         "phase_source": {"kind": "ck8", "bundle": bundle.name,
                                           "bundle_sha256": _sha256(bundle),
                                           "sourcefile": shard.name,
                                           "sourcefile_sha256": _sha256(shard),
                                           "token_read": "all49", "feature_layer": 12,
                                           "denoise_index": 3}})
        payload = {"X": xout, "gt_phase_code": gt_phase, "phase_code": phases}
        payload.update(fields)
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmpname = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=str(out.parent))
        os.close(fd)
        tmp = Path(tmpname)
        try:
            payload["meta_json"] = np.asarray(json.dumps(out_meta, ensure_ascii=False, sort_keys=True))
            with tmp.open("wb") as f:
                np.savez(f, **payload)
            tmp.replace(out)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
    return {"rows": n, "output": str(out), "slug": slug}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True, type=Path)
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    print(json.dumps(prepare(a.shard, a.bundle, a.slug, a.out, a.batch), ensure_ascii=False))


if __name__ == "__main__":
    main()
