"""Launch one extraction shard in the frozen-GR00T model container (same image/mounts as
groot_multitask_20261021/run_extract.py). Called through run_extract.sh (harness guard).

env: GPUS=<one gpu id>  HARNESS_RECEIPT (set by the guard)
args: [--smoke] [--shard k --num-shards n] [--work DIR]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from experiment import ADAPTER, REPO, STAGING_ROOT, WORK  # noqa: E402
from scripts.path_setup import CACHE_ROOT  # noqa: E402
sys.path.insert(0, str(ADAPTER))


def container_path(host: Path) -> str:
    return "/cache/" + str(host.resolve().relative_to(CACHE_ROOT.resolve()))


def build_cmd(*, gpu: int, smoke: bool, shard: int, num_shards: int, work: Path, image: str) -> list[str]:
    out = "smoke_latents" if smoke else "latents"
    manifest = "metadata/manifest_smoke.json" if smoke else "metadata/manifest.json"
    name = f"atomic65-qam-extract-s{shard}of{num_shards}-{os.getpid()}"
    cmd = ["docker", "run", "--rm", "--pull=never", "--log-driver", "none", "--name", name, "--hostname", "kanu",
           "--gpus", f"device={gpu}", "--network", "none", "--shm-size", "1g", "--tmpfs", "/tmp:rw,size=1g",
           "--entrypoint", "python", "-v", f"{REPO}:/temporal_vla:ro", "-v", f"{work}:/work:rw",
           "-v", f"{CACHE_ROOT}:/cache:ro",
           "-v", "/home/dongkyu/.cache/huggingface:/home/dongkyu/.cache/huggingface:ro",
           "-e", "HF_HOME=/cache/datasets/huggingface", "-e", "HF_HUB_OFFLINE=1", "-e", "TRANSFORMERS_OFFLINE=1",
           "-e", "HF_MODULES_CACHE=/tmp/hf_modules", "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "OMP_NUM_THREADS=4",
           "-e", "OPENBLAS_NUM_THREADS=4", "-e", "TOKENIZERS_PARALLELISM=false",
           "-e", f"PDK_LOG_FILE=/work/extraction_module_s{shard}.log",
           "-e", "PYTHONPATH=/temporal_vla/src/policies/Isaac-GR00T:/temporal_vla", "-w", "/temporal_vla", image,
           "scripts/rl2_vla/groot_fold27_20261028/common_qam/extract.py", "extract",
           "--manifest", f"/work/{manifest}", "--out", f"/work/{out}",
           "--package-root", container_path(STAGING_ROOT), "--checkpoint-root", "/cache/datasets",
           "--shard", str(shard), "--num-shards", str(num_shards)]
    if smoke:
        cmd.append("--smoke")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--work", type=Path, default=WORK)
    a = ap.parse_args()
    assert os.environ.get("HARNESS_RECEIPT"), "run through run_extract.sh (harness guard)"
    gpu = int(os.environ["GPUS"])
    assert (a.work / ("metadata/manifest_smoke.json" if a.smoke else "metadata/manifest.json")).exists()
    assert shutil.disk_usage(a.work).free > 20 * 1024 ** 3, "work filesystem reserve 20GiB"
    from paired_eval import MODEL_IMAGE
    cmd = build_cmd(gpu=gpu, smoke=a.smoke, shard=a.shard, num_shards=a.num_shards, work=a.work, image=MODEL_IMAGE)
    log = a.work / f"{'smoke_' if a.smoke else ''}extract_shard{a.shard}of{a.num_shards}.log"
    with log.open("a") as f:
        f.write("# " + " ".join(cmd) + "\n"); f.flush()
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    print("EXTRACTION_SHARD_COMPLETED", a.shard, a.num_shards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
