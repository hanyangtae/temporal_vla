"""
RoboCasa lerobot v2.1 데이터셋을 v3.0으로 일괄 변환.

xvla 컨테이너 내에서 실행 (lerobot >= 0.4 필요):
  docker compose --profile xvla run --rm xvla \
    python /temporal_vla/scripts/convert_v21_to_v30.py

옵션:
  --input-base   v2.1 데이터셋 루트 (기본: /temporal_vla/data/datasets/v1.0)
  --output-base  v3.0 출력 경로 (기본: /temporal_vla/data/datasets_v3/v1.0)
  --task-type    atomic / composite / 둘 다 (기본: 둘 다)
  --dry-run      변환 없이 발견된 데이터셋만 출력
"""

import argparse
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm


def find_datasets(base: Path, task_type: str | None = None) -> list[Path]:
    """v2.1 lerobot 데이터셋 경로 탐색."""
    if task_type:
        patterns = [f"{task_type}/*/*/lerobot"]
    else:
        patterns = ["atomic/*/*/lerobot", "composite/*/*/lerobot"]

    datasets = []
    for pattern in patterns:
        found = sorted(base.glob(pattern))
        valid = [p for p in found if (p / "meta" / "info.json").exists()]
        datasets.extend(valid)
    return datasets


def convert_single(src: Path, dst: Path) -> bool:
    """단일 데이터셋 v2.1 -> v3.0 변환."""
    dst.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m",
        "lerobot.datasets.v30.convert_dataset_v21_to_v30",
        f"--repo-id=local",
        f"--root={src}",
        f"--output-dir={dst}",
        "--push-to-hub=false",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="RoboCasa v2.1 -> v3.0 일괄 변환")
    parser.add_argument("--input-base", type=str, default="/temporal_vla/data/datasets/v1.0")
    parser.add_argument("--output-base", type=str, default="/temporal_vla/data/datasets_v3/v1.0")
    parser.add_argument("--split", type=str, default="pretrain", choices=["pretrain", "target"])
    parser.add_argument("--task-type", type=str, default=None, choices=["atomic", "composite"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src_base = Path(args.input_base) / args.split
    dst_base = Path(args.output_base) / args.split

    if not src_base.exists():
        print(f"Error: {src_base} does not exist")
        sys.exit(1)

    datasets = find_datasets(src_base, args.task_type)
    print(f"Found {len(datasets)} datasets under {src_base}")

    if args.dry_run:
        for ds in datasets:
            rel = ds.relative_to(src_base)
            print(f"  {rel}")
        return

    success, fail = 0, 0
    for ds in tqdm(datasets, desc="Converting"):
        rel = ds.relative_to(src_base)
        dst = dst_base / rel
        task_name = ds.parts[-3]
        print(f"\n[{task_name}] {rel}")

        if dst.exists() and (dst / "meta" / "info.json").exists():
            print("  Already converted, skipping.")
            success += 1
            continue

        if convert_single(ds, dst):
            success += 1
        else:
            fail += 1

    print(f"\nDone: {success} succeeded, {fail} failed out of {len(datasets)} total")


if __name__ == "__main__":
    main()
