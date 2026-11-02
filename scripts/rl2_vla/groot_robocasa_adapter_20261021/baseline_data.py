"""OpenDrawer QAM baseline: episode latent 추출 산출물의 계약·저장·검증 loader (numpy 전용).

추출 CLI (`extract_opendrawer_latents.py`, lerobot 컨테이너) 와 학습 CLI (`train_qam_baseline.py`,
host `.venvs/rl2_iid`) 가 공유한다. 정본 설명은 CONTRACT.md §11.

산출물 (추출 out 폴더):
- ``extraction_contract.json`` : 추출 계약 (O_EXCL 생성, 재실행 시 byte 동일해야 resume 허용)
- ``episodes/episode_<idx>.json`` : episode commit marker (O_EXCL). npz 지문·source 상대경로·sha 포함
- ``episodes/episode_<idx>__<sha16>.npz`` : 내용 지문이 파일명에 들어간 immutable array 묶음
- ``extraction_complete.json`` : 102 episode 전부 commit 된 뒤에만 생성 (O_EXCL)

규약
- 결정 시점 = 0, 5, 10, ... < T (H16 예측 / 5 실행). feature = 첫 denoise action_decoder 입력 last16 평균.
- expert target = 데이터셋 action 을 checkpoint metadata 로 정규화한 16-step chunk + validity mask.
  모델 예측으로 대체하지 않는다.
- reward = 데이터셋 ``next.reward`` 원본 그대로 (T 행 전부). done = ``next.done`` 원본.
- terminal 규약 (offline dataset-end): next.done 은 마지막 행에서만 True → 마지막 transition 이
  terminal (discount 0). 이것은 **offline 데이터 끝 규약**이며, 실제 물리 env 가 그 시점에 종료했는지는
  모른다 (historical converter done_mode 미복원). Octo/V-GPS 의 last-3 성공 규약은 쓰지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from composition import FEATURE_CAPTURE_POINT, FEATURE_DIM, SPEC
from contract import (
    ACTION_HORIZON,
    CHECKPOINT_SUBFOLDER,
    CONFIG_SHA256,
    ContractError,
    ENV_ACTION_DIM,
    FEATURE_INTERFACE,
    METADATA_SHA256,
    NUM_INFERENCE_STEPS,
)
from data_contract import (
    AUX_SPLIT_V2_SHA256,
    LOCAL_OPENDRAWER,
    OPENDRAWER_PACKAGE_REL,
)

EXTRACTION_VERSION = "groot_opendrawer_decoder_latent_extraction/v1"
CHECKPOINT_REPO = "robocasa/robocasa365_checkpoints"
CHECKPOINT_REVISION = "c484448aba1a9b60a04c9b0ca117241518ea69f3"
EXPECTED_FEATURE_SIGNATURE = "ccbfb51ca01724cb"
EXECUTE_STEPS = 5
GAMMA = 0.99
REWARD_POSITIVE_TAIL = 16          # 실측: 102 episode 전부 마지막 16 행 reward 1, 그 앞은 0
SEED_DERIVATION = "sha256(f'{base_seed}|{episode_id}|{decision_step}')[:4] big-endian & 0x7fffffff"

REWARD_CONVENTION = {
    "rewards": "dataset next.reward verbatim, all T rows (float32)",
    "done": "dataset next.done verbatim, all T rows",
    "offline_terminal_convention": "offline_dataset_end: next.done true only at the last row -> "
                                   "end_reason=terminal at T, final transition discount 0",
    "physical_env_termination": "unknown (historical converter done_mode not recovered); "
                                "dataset end is not claimed to be an observed env termination",
    "audit_required": f"last {REWARD_POSITIVE_TAIL} rows reward==1, all earlier rows reward==0, "
                      "next.done true only at the last row; otherwise refuse",
    "not_used": "Octo/V-GPS last-3 success relabel convention",
}

NPZ_KEYS = ("decision_steps", "decision_seeds", "latents", "decoder_calls", "expert_chunks",
            "chunk_valid", "raw_actions", "rewards", "done", "frame_index")


# ---------------------------------------------------------------------------
# 지문 / 원자적 쓰기
# ---------------------------------------------------------------------------

def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_path(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_exclusive(path: str | Path, data: bytes) -> None:
    """tmp 에 쓰고 fsync 한 뒤 link 로 생성한다. 대상이 이미 있으면 FileExistsError (덮어쓰기 없음)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()


def write_json_exclusive(path: str | Path, obj: Any) -> None:
    write_exclusive(path, json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False).encode())


def ensure_same_or_create(path: str | Path, obj: Any, *, what: str) -> None:
    """파일이 없으면 O_EXCL 생성, 있으면 내용이 같아야 한다 (다르면 거부, 덮어쓰지 않음)."""
    path = Path(path)
    if path.exists():
        got = json.loads(path.read_text())
        if canonical_json(got) != canonical_json(obj):
            raise ContractError(f"{what} 충돌: 기존 {path} 와 현재 계약이 다름 — 새 출력 폴더를 쓸 것")
        return
    write_json_exclusive(path, obj)


# ---------------------------------------------------------------------------
# 추출 계약
# ---------------------------------------------------------------------------

def decision_steps(length: int, execute_steps: int = EXECUTE_STEPS) -> np.ndarray:
    if length <= 0:
        raise ContractError(f"episode length {length}")
    return np.arange(0, int(length), int(execute_steps), dtype=np.int64)


def decision_seed(base_seed: int, episode_id: str, step: int) -> int:
    h = hashlib.sha256(f"{int(base_seed)}|{episode_id}|{int(step)}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def extraction_contract(*, base_seed: int, aux_split_sha256: str, code_sha256: Mapping[str, str],
                        profile_rel: str, profile_sha256: str, image_id: str,
                        purpose: str = "published_package_subset_for_baseline") -> dict:
    if aux_split_sha256 != AUX_SPLIT_V2_SHA256:
        raise ContractError(f"aux split sha {aux_split_sha256} != v2 {AUX_SPLIT_V2_SHA256}")
    if SPEC.signature() != EXPECTED_FEATURE_SIGNATURE:
        raise ContractError(f"feature signature {SPEC.signature()} != {EXPECTED_FEATURE_SIGNATURE}")
    return {
        "version": EXTRACTION_VERSION,
        "purpose": purpose,
        "historical_membership_claimed": False,
        "feature": {"signature": SPEC.signature(), "interface": FEATURE_INTERFACE,
                    "capture_point": FEATURE_CAPTURE_POINT, "token_selection": "last16_action_tokens",
                    "pool": "mean", "denoise_step": 0, "dim": FEATURE_DIM,
                    "decoder_calls_per_decision": NUM_INFERENCE_STEPS},
        "checkpoint": {"repo": CHECKPOINT_REPO, "revision": CHECKPOINT_REVISION,
                       "subfolder": CHECKPOINT_SUBFOLDER, "config_sha256": CONFIG_SHA256,
                       "metadata_sha256": METADATA_SHA256},
        "dataset": {"package_rel": OPENDRAWER_PACKAGE_REL, "info_sha256": LOCAL_OPENDRAWER["info_sha256"],
                    "episodes_jsonl_sha256": LOCAL_OPENDRAWER["episodes_jsonl_sha256"],
                    "episodes": LOCAL_OPENDRAWER["episodes"]},
        "aux_split_sha256": aux_split_sha256,
        "decision": {"horizon": ACTION_HORIZON, "execute_steps": EXECUTE_STEPS,
                     "grid": "0, execute_steps, 2*execute_steps, ... < T"},
        "seed": {"base_seed": int(base_seed), "derivation": SEED_DERIVATION,
                 "applied_via": "scripts/serve/lerobot.py::_apply_inference_seed (random/np/torch/cuda)"},
        "preprocessing": {"profile_rel": profile_rel, "profile_sha256": profile_sha256,
                          "path": "serve parse_payload -> _apply_input_remap -> GrootPolicyAdapter "
                                  "preprocessor -> GrootPolicy.predict_action_chunk",
                          "image_transport": "lossless PNG base64 (decoded by serve parse_payload)"},
        "action_target": {"source": "expert_dataset_action", "normalization": "checkpoint_metadata",
                          "layout": "(n, 16, 12) model order normalized + chunk_valid mask"},
        "reward_convention": REWARD_CONVENTION,
        "code_sha256": dict(sorted(code_sha256.items())),
        "runtime_image_id": image_id,
    }


def contract_sha256(contract: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(contract))


# ---------------------------------------------------------------------------
# reward audit / episode record
# ---------------------------------------------------------------------------

def audit_rewards(rewards: np.ndarray, done: np.ndarray) -> dict:
    """데이터셋 reward/done 실측 패턴 확인. 패턴이 다르면 거부한다 (규약 문서가 이 패턴 전제)."""
    r = np.asarray(rewards)
    d = np.asarray(done, dtype=bool)
    T = len(r)
    if r.ndim != 1 or d.shape != (T,):
        raise ContractError(f"reward/done shape {r.shape} {d.shape}")
    if not np.all(np.isfinite(r)):
        raise ContractError("reward 비유한")
    k = REWARD_POSITIVE_TAIL
    if T <= k:
        raise ContractError(f"episode length {T} <= {k}")
    tail_ok = bool(np.all(r[-k:] == 1.0))
    head_ok = bool(np.all(r[:-k] == 0.0))
    done_idx = np.flatnonzero(d).tolist()
    done_ok = done_idx == [T - 1]
    if not (tail_ok and head_ok and done_ok):
        raise ContractError(f"reward audit 실패: tail16==1 {tail_ok}, earlier==0 {head_ok}, "
                            f"done indices {done_idx[:5]} (want [{T - 1}])")
    return {"length": T, "positive_steps": int(np.sum(r > 0)), "first_positive": int(T - k),
            "done_indices": done_idx, "last16_all_one": tail_ok, "earlier_all_zero": head_ok}


def build_episode_arrays(*, episode_id: str, latents: np.ndarray, decoder_calls: Sequence[int],
                         seeds: Sequence[int], raw_actions: np.ndarray, rewards: np.ndarray,
                         done: np.ndarray, frame_index: np.ndarray, codec, raw_index: np.ndarray,
                         base_seed: int) -> dict:
    """추출 결과 → npz array dict. 결정 grid·seed·target 을 다시 계산해 정렬을 확인한다."""
    from transitions import expert_chunk_targets

    raw = np.asarray(raw_actions, dtype=np.float64)
    T = len(raw)
    if raw.shape != (T, ENV_ACTION_DIM):
        raise ContractError(f"{episode_id}: raw action shape {raw.shape}")
    fi = np.asarray(frame_index, dtype=np.int64)
    if not np.array_equal(fi, np.arange(T)):
        raise ContractError(f"{episode_id}: frame_index 가 0..T-1 연속이 아님")
    steps = decision_steps(T)
    lat = np.asarray(latents, dtype=np.float32)
    if lat.shape != (len(steps), FEATURE_DIM) or not np.all(np.isfinite(lat)):
        raise ContractError(f"{episode_id}: latents {lat.shape} != ({len(steps)}, {FEATURE_DIM}) 또는 비유한")
    calls = np.asarray(decoder_calls, dtype=np.int64)
    if calls.shape != steps.shape or not np.all(calls == NUM_INFERENCE_STEPS):
        raise ContractError(f"{episode_id}: decoder 호출 수 != {NUM_INFERENCE_STEPS}")
    sd = np.asarray(seeds, dtype=np.int64)
    want = np.asarray([decision_seed(base_seed, episode_id, t) for t in steps], dtype=np.int64)
    if not np.array_equal(sd, want):
        raise ContractError(f"{episode_id}: decision seed 가 결정적 유도값과 다름")
    audit_rewards(rewards, done)
    chunks, valid = expert_chunk_targets(raw, steps, codec=codec, raw_index=raw_index)
    return {"decision_steps": steps, "decision_seeds": sd, "latents": lat, "decoder_calls": calls,
            "expert_chunks": chunks.astype(np.float32), "chunk_valid": valid, "raw_actions": raw,
            "rewards": np.asarray(rewards, dtype=np.float32), "done": np.asarray(done, dtype=bool),
            "frame_index": fi}


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    import io

    if set(arrays) != set(NPZ_KEYS):
        raise ContractError(f"npz key {sorted(arrays)} != {sorted(NPZ_KEYS)}")
    buf = io.BytesIO()
    np.savez(buf, **{k: arrays[k] for k in NPZ_KEYS})
    return buf.getvalue()


def episode_json_name(episode_index: int) -> str:
    return f"episode_{int(episode_index):06d}.json"


def commit_episode(out_dir: str | Path, *, episode_index: int, arrays: Mapping[str, np.ndarray],
                   meta: Mapping[str, Any], contract: Mapping[str, Any]) -> dict:
    """npz (내용 지문 파일명) → json commit marker 순으로 O_EXCL 생성. 기존 commit 이 있으면 거부."""
    ep_dir = Path(out_dir) / "episodes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    jpath = ep_dir / episode_json_name(episode_index)
    if jpath.exists():
        raise ContractError(f"{jpath} 이미 commit 됨 — 덮어쓰기 금지 (verify_committed_episode 로 확인)")
    data = _npz_bytes(arrays)
    sha = sha256_bytes(data)
    npz_name = f"episode_{int(episode_index):06d}__{sha[:16]}.npz"
    npath = ep_dir / npz_name
    if npath.exists():
        if sha256_path(npath) != sha:
            raise ContractError(f"{npath} 내용 지문 불일치 (파일명 sha16 와 다름)")
    else:
        write_exclusive(npath, data)
    rec = dict(meta)
    rec.update({"episode_index": int(episode_index), "contract_sha256": contract_sha256(contract),
                "npz": {"file": npz_name, "sha256": sha, "bytes": len(data)},
                "n_decisions": int(len(arrays["decision_steps"])), "length": int(len(arrays["rewards"]))})
    write_json_exclusive(jpath, rec)
    return rec


def verify_committed_episode(out_dir: str | Path, episode_index: int, contract: Mapping[str, Any],
                             *, expect: Mapping[str, Any] | None = None) -> dict | None:
    """commit 된 episode 를 검증한다. 없으면 None. 계약·지문·split 이 다르면 거부 (resume 금지)."""
    ep_dir = Path(out_dir) / "episodes"
    jpath = ep_dir / episode_json_name(episode_index)
    if not jpath.exists():
        return None
    rec = json.loads(jpath.read_text())
    if rec.get("contract_sha256") != contract_sha256(contract):
        raise ContractError(f"{jpath}: 추출 계약 지문 불일치 — resume 거부")
    npath = ep_dir / rec["npz"]["file"]
    if not npath.is_file() or sha256_path(npath) != rec["npz"]["sha256"]:
        raise ContractError(f"{npath}: npz 없음 또는 sha256 불일치 — resume 거부")
    for k, v in (expect or {}).items():
        if rec.get(k) != v:
            raise ContractError(f"{jpath}: {k}={rec.get(k)!r} != 기대 {v!r} — resume 거부")
    return rec


def load_episode_arrays(out_dir: str | Path, rec: Mapping[str, Any]) -> dict:
    with np.load(Path(out_dir) / "episodes" / rec["npz"]["file"], allow_pickle=False) as z:
        if set(z.files) != set(NPZ_KEYS):
            raise ContractError(f"{rec['npz']['file']}: npz key {sorted(z.files)}")
        return {k: z[k] for k in NPZ_KEYS}


# ---------------------------------------------------------------------------
# loader (학습 측)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LoadedExtraction:
    train: list
    val: list
    receipt: dict


def _read_parquet_columns(path: Path) -> dict:
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["action", "next.reward", "next.done", "frame_index"]).to_pydict()
    return {"raw_actions": np.asarray(t["action"], dtype=np.float64),
            "rewards": np.asarray(t["next.reward"], dtype=np.float32).reshape(-1),
            "done": np.asarray(t["next.done"], dtype=bool).reshape(-1),
            "frame_index": np.asarray(t["frame_index"], dtype=np.int64).reshape(-1)}


def load_extraction(extraction_dir: str | Path, split_path: str | Path, *, codec, raw_index: np.ndarray,
                    data_root: str | Path | None = None, verify_source: bool = True,
                    allow_subset: bool = False) -> LoadedExtraction:
    """저장된 추출을 검증해 train/val EpisodeInput 으로 돌려준다.

    검사: split 파일 sha = v2 고정값 + 독립 재계산 (validate_aux_split), 추출 계약 (feature signature,
    checkpoint, split sha, 결정 grid), episode 별 npz sha·contract 지문·split 일치, seed 재유도,
    reward audit, expert target 재계산, (verify_source) parquet 원본 action/reward/done 과 byte 대조.
    allow_subset=False 면 102 episode (92/10) 가 전부 있어야 한다.
    """
    from data_contract import validate_aux_split
    from transitions import EpisodeInput, expert_chunk_targets

    ext = Path(extraction_dir)
    split_sha = sha256_path(split_path)
    if split_sha != AUX_SPLIT_V2_SHA256:
        raise ContractError(f"split 파일 sha {split_sha} != v2 {AUX_SPLIT_V2_SHA256}")
    aux = validate_aux_split(split_path, data_root=data_root, verify_files=verify_source)
    split_of = {e["episode_id"]: e["split"] for e in aux["episodes"]}
    split_doc = json.loads(Path(split_path).read_text())
    index_of = {e["episode_id"]: int(e["episode_index"]) for e in split_doc["episodes"]}
    parquet_sha = {e["episode_id"]: e["parquet_sha256"] for e in split_doc["episodes"]}

    contract = json.loads((ext / "extraction_contract.json").read_text())
    if contract.get("version") != EXTRACTION_VERSION:
        raise ContractError(f"추출 계약 version {contract.get('version')!r}")
    checks = {
        "feature.signature": (contract["feature"]["signature"], SPEC.signature()),
        "feature.capture_point": (contract["feature"]["capture_point"], FEATURE_CAPTURE_POINT),
        "feature.denoise_step": (contract["feature"]["denoise_step"], 0),
        "checkpoint.revision": (contract["checkpoint"]["revision"], CHECKPOINT_REVISION),
        "checkpoint.config_sha256": (contract["checkpoint"]["config_sha256"], CONFIG_SHA256),
        "checkpoint.metadata_sha256": (contract["checkpoint"]["metadata_sha256"], METADATA_SHA256),
        "aux_split_sha256": (contract["aux_split_sha256"], split_sha),
        "decision.execute_steps": (contract["decision"]["execute_steps"], EXECUTE_STEPS),
        "decision.horizon": (contract["decision"]["horizon"], ACTION_HORIZON),
        "historical_membership_claimed": (contract.get("historical_membership_claimed"), False),
    }
    bad = {k: got for k, (got, want) in checks.items() if got != want}
    if bad:
        raise ContractError(f"추출 계약 불일치: {bad}")
    base_seed = int(contract["seed"]["base_seed"])
    pkg = None
    if verify_source:
        if data_root is None:
            from scripts.path_setup import DATA_ROOT

            data_root = DATA_ROOT
        pkg = Path(data_root) / OPENDRAWER_PACKAGE_REL

    records = sorted((ext / "episodes").glob("episode_*.json"))
    found: dict[str, list] = {"train": [], "val": []}
    seen = set()
    per_episode = []
    for jp in records:
        idx = int(jp.stem.split("_")[1])
        rec = verify_committed_episode(ext, idx, contract)
        eid = rec["episode_id"]
        if eid not in split_of or index_of[eid] != idx:
            raise ContractError(f"{jp}: episode {eid}/{idx} 가 split 파일에 없음")
        if rec.get("split") != split_of[eid]:
            raise ContractError(f"{eid}: 추출 split {rec.get('split')!r} != split 파일 {split_of[eid]!r} (누수 방지)")
        if rec["sources"]["parquet"]["sha256"] != parquet_sha[eid]:
            raise ContractError(f"{eid}: 추출 source parquet sha 가 split 파일과 다름")
        a = load_episode_arrays(ext, rec)
        T = len(a["rewards"])
        steps = decision_steps(T)
        if not np.array_equal(a["decision_steps"], steps):
            raise ContractError(f"{eid}: decision grid 불일치")
        if not np.array_equal(a["decision_seeds"],
                              [decision_seed(base_seed, eid, t) for t in steps]):
            raise ContractError(f"{eid}: seed 재유도 불일치")
        if a["latents"].shape != (len(steps), FEATURE_DIM) or not np.all(np.isfinite(a["latents"])):
            raise ContractError(f"{eid}: latents shape/유한성")
        if not np.all(a["decoder_calls"] == NUM_INFERENCE_STEPS):
            raise ContractError(f"{eid}: decoder 호출 수")
        audit_rewards(a["rewards"], a["done"])
        chunks, valid = expert_chunk_targets(a["raw_actions"], steps, codec=codec, raw_index=raw_index)
        if not (np.array_equal(chunks.astype(np.float32), a["expert_chunks"]) and
                np.array_equal(valid, a["chunk_valid"])):
            raise ContractError(f"{eid}: expert target 이 데이터셋 action 재계산과 다름")
        if pkg is not None:
            src = pkg / rec["sources"]["parquet"]["rel"]
            if sha256_path(src) != parquet_sha[eid]:
                raise ContractError(f"{eid}: source parquet sha 불일치")
            cols = _read_parquet_columns(src)
            for k in ("raw_actions", "rewards", "done", "frame_index"):
                if not np.array_equal(cols[k], a[k]):
                    raise ContractError(f"{eid}: 저장된 {k} 가 parquet 원본과 다름 (verbatim 위반)")
        found[rec["split"]].append(EpisodeInput(
            episode_id=eid, run_group=eid, split=rec["split"], length=T, end_reason="terminal",
            decision_steps=steps, latents=a["latents"], action_chunks=a["expert_chunks"],
            rewards=a["rewards"].astype(np.float64), chunk_valid=a["chunk_valid"], final_latent=None))
        seen.add(eid)
        per_episode.append({"episode_id": eid, "split": rec["split"], "npz_sha256": rec["npz"]["sha256"],
                            "n_decisions": int(len(steps)), "length": int(T)})

    train_ids = {e.episode_id for e in found["train"]}
    val_ids = {e.episode_id for e in found["val"]}
    if train_ids & val_ids:
        raise ContractError("train/val episode 겹침")
    want_train = {k for k, s in split_of.items() if s == "train"}
    want_val = {k for k, s in split_of.items() if s == "val"}
    if not allow_subset and (train_ids != want_train or val_ids != want_val):
        raise ContractError(f"추출 episode 가 split 전체와 다름: train {len(train_ids)}/{len(want_train)}, "
                            f"val {len(val_ids)}/{len(want_val)} (smoke 는 allow_subset)")
    if not found["train"] or not found["val"]:
        raise ContractError("train 또는 val episode 없음")
    complete = ext / "extraction_complete.json"
    receipt = {"extraction_contract_sha256": contract_sha256(contract), "aux_split_sha256": split_sha,
               "train_episodes": len(found["train"]), "val_episodes": len(found["val"]),
               "subset": bool(train_ids != want_train or val_ids != want_val),
               "extraction_complete_sha256": sha256_path(complete) if complete.is_file() else None,
               "source_verified": bool(verify_source), "episodes": per_episode,
               "feature_signature": SPEC.signature()}
    return LoadedExtraction(found["train"], found["val"], receipt)


def episode_index_list(spec: str | None, available: Iterable[int]) -> list[int]:
    """'0,1,5' 또는 '0-9' 형식 → 정렬된 index. None 이면 전부."""
    avail = sorted(set(int(i) for i in available))
    if not spec:
        return avail
    out = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    bad = sorted(out - set(avail))
    if bad:
        raise ContractError(f"없는 episode index {bad[:5]}")
    return sorted(out)
