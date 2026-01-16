#!/usr/bin/env python3
"""
LeRobot 로컬 포맷을 HuggingFace Dataset으로 변환

Usage:
    python scripts/convert_to_hf_dataset.py
"""

import json
from pathlib import Path
from datasets import Dataset, Features, Value, Sequence, Image, DatasetDict
from tqdm import tqdm

def main():
    output_dir = Path('data/datasets/lerobot_drawer_time')
    hf_output_dir = Path('data/datasets/hf_drawer_time')

    # episodes.jsonl 읽기
    print('📖 Loading episodes.jsonl...')
    episodes = []
    with open(output_dir / 'episodes.jsonl') as f:
        for line in f:
            episodes.append(json.loads(line))

    print(f'   Loaded {len(episodes)} frames')

    # 데이터 준비
    print('🔧 Converting to HuggingFace format...')

    data = {
        'episode_index': [],
        'frame_index': [],
        'timestamp': [],
        'index': [],
        'action': [],
        'observation.state': [],
        'observation.images.agentview': [],
        'observation.images.eye_in_hand': [],
        'task': [],
        'time_to_go': [],
    }

    for item in tqdm(episodes, desc="Processing"):
        data['episode_index'].append(item['episode_index'])
        data['frame_index'].append(item['frame_index'])
        data['timestamp'].append(item['timestamp'])
        data['index'].append(item['index'])
        data['action'].append(item['action'])
        data['observation.state'].append(item['observation.state'])
        data['observation.images.agentview'].append(str(output_dir / item['observation.images.agentview']))
        data['observation.images.eye_in_hand'].append(str(output_dir / item['observation.images.eye_in_hand']))
        data['task'].append(item['task'])
        data['time_to_go'].append(item['time_to_go'])

    # Dataset 생성
    print('📦 Creating Dataset...')
    features = Features({
        'episode_index': Value('int64'),
        'frame_index': Value('int64'),
        'timestamp': Value('float64'),
        'index': Value('int64'),
        'action': Sequence(Value('float64')),
        'observation.state': Sequence(Value('float64')),
        'observation.images.agentview': Image(),
        'observation.images.eye_in_hand': Image(),
        'task': Value('string'),
        'time_to_go': Value('float64'),
    })

    ds = Dataset.from_dict(data, features=features)
    ds_dict = DatasetDict({'train': ds})

    print(f'✅ Dataset created: {ds_dict}')

    # 저장
    print(f'💾 Saving to {hf_output_dir}...')
    hf_output_dir.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(hf_output_dir))
    print('✅ Done!')

    # 검증
    print('\n🔍 Verification:')
    loaded = DatasetDict.load_from_disk(str(hf_output_dir))
    print(f'   Loaded: {loaded}')
    print(f'   Sample time_to_go: {loaded["train"][0]["time_to_go"]:.2f}s')


if __name__ == "__main__":
    main()
