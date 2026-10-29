#!/usr/bin/env bash
# Read-only inventory of the storage host (승준), run from its git checkout by atomic65_stage.py:
#   remote_compute.sh run "nice -n 10 bash scripts/rl2_vla/atomic65_stage_20261028/remote_inventory.sh <args>"
# Writes nothing. Prints one JSON document between INVENTORY_BEGIN/INVENTORY_END.
#
# Args: SCRIPT_SHA DATASETS_DIR SCAN_MAXDEPTH ALIAS=ROOT [ALIAS=ROOT ...]
#   every ALIAS=ROOT is scanned for v1.0/pretrain/atomic/<Task>/<date>/lerobot packages,
#   receipts/, and archives/**/*.tar(.part); DATASETS_DIR additionally gets df, a
#   top-level listing and a depth-bounded search for any other atomic lerobot package.
set -Eeuo pipefail

main() {
  local SCRIPT_SHA="$1" DATASETS="$2" DEPTH="$3"; shift 3
  local self_sha; self_sha=$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)
  [[ "$self_sha" == "$SCRIPT_SHA" ]] || { echo "STATUS CODE_MISMATCH"; exit 66; }
  command -v python3 >/dev/null || { echo "STATUS MISSING_TOOL python3"; exit 65; }
  echo "INVENTORY_BEGIN"
  python3 - "$DATASETS" "$DEPTH" "$@" <<'PY'
import json, os, shutil, subprocess, sys
from pathlib import Path

datasets, depth, roots = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]

def package(pkg):
    rec = {"rel": str(pkg), "files": 0, "bytes": 0}
    for p in pkg.rglob("*"):
        if p.is_file():
            rec["files"] += 1; rec["bytes"] += p.stat().st_size
    try:
        info = json.loads((pkg / "meta/info.json").read_text())
        rec["info_total_episodes"] = info.get("total_episodes")
        rec["info_total_frames"] = info.get("total_frames")
    except Exception as e:  # report, do not hide
        rec["info_error"] = repr(e)
    try:
        lines = [l for l in (pkg / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
        rec["episodes_jsonl_rows"] = len(lines)
    except Exception as e:
        rec["episodes_error"] = repr(e)
    rec["parquet_files"] = sum(1 for _ in (pkg / "data").rglob("*.parquet")) if (pkg / "data").is_dir() else 0
    rec["video_files"] = sum(1 for _ in (pkg / "videos").rglob("*.mp4")) if (pkg / "videos").is_dir() else 0
    return rec

out = {"datasets_dir_exists": datasets.is_dir(), "roots": {}}
if datasets.is_dir():
    du = shutil.disk_usage(datasets)
    out["disk"] = {"total": du.total, "used": du.used, "free": du.free}
    out["datasets_top_level"] = sorted(p.name for p in datasets.iterdir())
    # any atomic lerobot package anywhere under the datasets dir (depth bounded)
    found = subprocess.run(
        ["find", str(datasets), "-maxdepth", str(depth), "-type", "d", "-name", "lerobot",
         "-path", "*/pretrain/atomic/*"],
        capture_output=True, text=True, timeout=900)
    out["atomic_lerobot_dirs_anywhere"] = sorted(
        os.path.relpath(l, datasets) for l in found.stdout.splitlines() if l)
    out["find_returncode"] = found.returncode
for spec in roots:
    alias, root = spec.split("=", 1)
    root = Path(root); r = {"exists": root.is_dir(), "packages": {}, "receipts": {},
                            "archives": {}, "parts": {}}
    if root.is_dir():
        base = root / "v1.0/pretrain/atomic"
        for pkg in sorted(base.glob("*/*/lerobot")) if base.is_dir() else []:
            rec = package(pkg); rec["rel"] = str(pkg.relative_to(root))
            r["packages"].setdefault(pkg.parent.parent.name, []).append(rec)
        for rc in sorted((root / "receipts").glob("*.json")) if (root / "receipts").is_dir() else []:
            try:
                j = json.loads(rc.read_text())
                r["receipts"][rc.stem] = {k: j.get(k) for k in (
                    "status", "archive_sha256", "archive_bytes", "package_manifest_sha256",
                    "episodes_total", "frames_total", "episodes_outside_filter_n")}
            except Exception as e:
                r["receipts"][rc.stem] = {"error": repr(e)}
        arch = root / "archives"
        if arch.is_dir():
            for p in sorted(arch.rglob("*.tar")):
                r["archives"][str(p.relative_to(root))] = p.stat().st_size
            for p in sorted(arch.rglob("*.tar.part")):
                r["parts"][str(p.relative_to(root))] = p.stat().st_size
    out["roots"][alias] = r
print(json.dumps(out))
PY
  echo "INVENTORY_END"
}

main "$@" </dev/null
exit $?
