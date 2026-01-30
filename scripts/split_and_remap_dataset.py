import os
import logging
import multiprocessing as mp
from datasets import load_from_disk

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# CPU 코어 수 자동 감지 (최대 16코어까지 사용)
NUM_PROC = min(mp.cpu_count(), 16)
logger.info(f"🚀 Using {NUM_PROC} CPU cores for parallel processing")

def main():
    # 경로 설정
    # 사용자는 data/libero_hf 라고 했지만 실제 구조는 data/libero_hf/dataset 임을 확인했으므로 이를 반영
    # 만약 load_from_disk가 실패하면 경로를 다시 확인해야 함
    input_path = "data/libero_hf/dataset" 
    output_path = "data/libero_hf_split/libero_goal"
    
    # 매핑 정보 (scripts/convert_rel_to_abs.py 234-237 참조)
    goal_task_map = {
        10: 8, 11: 9, 12: 3, 13: 6, 14: 2,
        15: 5, 16: 7, 17: 1, 18: 4, 19: 0
    }
    target_indices = list(goal_task_map.keys())

    logger.info(f"📂 Loading dataset from {input_path}")
    if not os.path.exists(input_path):
        # 만약 dataset 폴더가 없다면 부모 폴더를 시도해본다
        if os.path.exists("data/libero_hf"):
             logger.warning(f"⚠️ {input_path} not found. Trying data/libero_hf")
             input_path = "data/libero_hf"
        else:
            logger.error(f"❌ Input path does not exist: {input_path}")
            return

    try:
        # 메모리 효율적인 로딩 (대용량 데이터셋 대비)
        ds = load_from_disk(input_path, keep_in_memory=False)
        
        # DatasetDict인 경우 처리 (train/test split 등이 있는 경우)
        if hasattr(ds, 'keys') and 'train' in ds.keys():
            logger.info(f"Dataset keys: {ds.keys()}")
            ds = ds['train']
            logger.info("Selected 'train' split.")
        
        logger.info(f"📊 Original dataset size: {len(ds)}")

        # 1. Filter (병렬 처리 + 배치 처리)
        logger.info(f"🔍 Filtering tasks: {target_indices}")
        target_set = set(target_indices)  # lookup 속도 향상을 위해 set 사용
        
        def filter_fn(examples):
            # 배치 단위로 처리
            return [idx in target_set for idx in examples['task_index']]
        
        filtered_ds = ds.filter(
            filter_fn, 
            batched=True, 
            num_proc=NUM_PROC,
            desc="Filtering tasks"
        )
        logger.info(f"✅ Filtered dataset size: {len(filtered_ds)}")

        if len(filtered_ds) == 0:
            logger.warning("⚠️ No data found for target tasks!")
            return

        # 2. Remap task_index (배치 처리로 성능 향상)
        logger.info("🔄 Remapping task indices...")
        
        def remap_task_index_batched(examples):
            # 배치 단위로 처리하여 성능 향상
            examples['task_index'] = [
                goal_task_map.get(old_idx, old_idx) 
                for old_idx in examples['task_index']
            ]
            return examples

        remapped_ds = filtered_ds.map(
            remap_task_index_batched, 
            batched=True,
            batch_size=1000,  # 배치 크기 설정
            num_proc=NUM_PROC, 
            desc="Remapping indices"
        )

        # 3. Save
        logger.info(f"💾 Saving to {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        remapped_ds.save_to_disk(output_path)
        logger.info("✅ Done! Dataset successfully processed and saved.")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
