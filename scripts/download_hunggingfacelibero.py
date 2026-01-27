#!/usr/bin/env python3
"""
HuggingFace에서 LIBERO 데이터셋을 다운로드하는 스크립트
"""

import os
import argparse
from pathlib import Path


def download_libero_dataset(output_dir: str = "data/libero_hf"):
    """
    HuggingFace에서 LIBERO 데이터셋을 다운로드합니다.
    
    Args:
        output_dir: 데이터셋을 저장할 경로
        subset: 특정 subset만 다운로드 (None이면 전체)
    """
    from datasets import load_dataset
    
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 LIBERO 데이터셋 다운로드 시작...")
    print(f"📂 저장 경로: {output_path.absolute()}")
    
    try:
        # 데이터셋 로드
        ds = load_dataset("HuggingFaceVLA/libero")
        
        # 데이터셋 정보 출력
        print(f"\n✅ 데이터셋 로드 완료!")
        print(f"📊 데이터셋 구조:")
        print(ds)
        
        # 로컬에 저장
        save_path = output_path / "dataset"
        print(f"\n💾 데이터셋 저장 중: {save_path}")
        ds.save_to_disk(str(save_path))
        
        print(f"\n🎉 다운로드 완료!")
        print(f"📂 저장 위치: {save_path.absolute()}")
        
        return ds
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="HuggingFace에서 LIBERO 데이터셋 다운로드"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/libero_hf",
        help="데이터셋 저장 경로 (기본값: data/libero_hf)"
    )
    
    args = parser.parse_args()
    
    download_libero_dataset(
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
