# DreamVLA RoboCasa Auxiliary Loss Training — 핸드오프 문서

> 이 문서는 다음 세션에서 학습을 시작하기 위한 임시 컨텍스트입니다. 학습 완료 후 삭제해도 됩니다.

## 현재 상태 (2026-03-24)

### 완료된 작업
1. **pretrain 데이터 다운로드**: robocasa 컨테이너에서 PickPlaceSinkToCounter pretrain 데이터 다운로드 완료
2. **LeRobot v2.1 → v3.0 변환 완료**: 108 에피소드, 26,397 프레임
   - 경로: `/temporal_vla/src/benchmarks/robocasa/datasets/v1.0/pretrain/atomic/PickPlaceSinkToCounter/20250819/lerobot`
3. **SAM feature 추출 완료**: 26,397 × 2 카메라 (rgb_static, rgb_gripper)
   - 경로: `/temporal_vla/src/benchmarks/robocasa/datasets/features/sam/`
4. **CoTracker trajectory 추출 완료**: 26,397 × 2 카메라
   - 경로: `/temporal_vla/src/benchmarks/robocasa/datasets/features/tracks/`

### 컨테이너 구성
- **dreamvla** (학습용): 물리 GPU 2 (컨테이너 내부 GPU 0), UUID `GPU-b66ae2c7`
- **dreamvla-eval** (eval용): 물리 GPU 3, UUID `GPU-9fdab52c` — 현재 내려가있을 수 있음
- **robocasa**: 물리 GPU 2 (`GPU-b66ae2c7`)

### CoTracker 추출 완료 확인 방법
```bash
# 각 카메라 26,397개 파일이 있어야 함
docker exec dreamvla bash -c "ls /temporal_vla/src/benchmarks/robocasa/datasets/features/tracks/rgb_static/training/ | wc -l"
docker exec dreamvla bash -c "ls /temporal_vla/src/benchmarks/robocasa/datasets/features/tracks/rgb_gripper/training/ | wc -l"
```

### cotracker 설치 참고
- dreamvla 학습 컨테이너에 `pip install git+https://github.com/facebookresearch/co-tracker.git` (v3.0)으로 설치됨
- `cotracker3_offline.py`에서 `.view()` → `.reshape()` 패치 적용됨 (공식 known issue)
- segment-anything도 pip으로 설치됨

## 학습 실행 방법

### 학습 스크립트
```bash
docker compose exec dreamvla bash scripts/train_dreamvla_robocasa.sh
```
> 주의: `docker compose exec dreamvla`는 학습용 컨테이너가 아닐 수 있음. `docker exec dreamvla`로 직접 지정 필요.

### 학습 스크립트 내용 (`scripts/train_dreamvla_robocasa.sh`)
핵심 인자:
- `--robocasa_dataset`: pretrain 데이터 경로
- `--sam_features_path`: SAM feature 경로 (`/temporal_vla/src/benchmarks/robocasa/datasets/features/sam`)
- `--track_label_path`: CoTracker track 경로 (`/temporal_vla/src/benchmarks/robocasa/datasets/features/tracks`)
- Auxiliary loss 플래그: `--obs_pred`, `--loss_image`, `--depth_pred`, `--loss_depth`, `--use_dit_head`, `--sam_feat_pred`, `--loss_sam_feat`, `--load_sam_features`, `--trajectory_pred`, `--loss_trajectory`, `--load_track_labels`, `--track_label_patch_size 8`, `--flow_as_mask`

### DreamVLA 13-tuple collator 형식
```
[0] images_primary    [1] text           [2] actions
[3] images_wrist      [4] states         [5] robot_obs
[6] depth_static      [7] depth_wrist    [8] dino_static
[9] dino_wrist        [10] sam_static    [11] sam_wrist
[12] track_dict
```

### Loss 구성요소
- `loss_arm_action` (Smooth L1) + `loss_gripper_action` (BCE) → 메인 action loss
- `loss_image` (MSE, weight 0.1) — `--obs_pred`, `--flow_as_mask`로 dynamic region 마스킹
- `loss_pred_depth` (SiLogLoss, weight 0.001) — `--depth_pred`
- `loss_pred_trajectory` (MSE, weight 0.1) — `--trajectory_pred`, `--load_track_labels`
- `loss_pred_sam_feat` (cosine, weight 0.01) — `--sam_feat_pred`, `--load_sam_features`
- `loss_pred_dino_feat` (cosine, weight 0.01) — 이번에는 안 씀

## 데이터 매칭 — 확인 필요
- SAM/CoTracker 파일명 = LeRobotDataset의 **global frame index** (`0.pt`, `1.npz`, ...)
- adapter의 `__getitem__(idx)`에서 `frame_indices = range(idx, idx + total_len)`으로 로딩
- 에피소드 경계 패딩은 `_is_pad` 체크로 걸러짐

### 잠재적 이슈: delta_timestamps vs 순차 인덱스
- LeRobotDataset에 `delta_timestamps`를 설정하면, `__getitem__(idx)`가 반환하는 이미지는 **타임스탬프 기반 매칭**으로 가져옴
- 그런데 SAM/track 로딩(`src/datasets/adapters/dreamvla.py:210`)은 `range(idx, idx + total_len)`으로 **순차 인덱스**를 사용
- 만약 fps가 정확히 일치하면 문제 없음 (delta_ts = `[0/fps, 1/fps, ...]` → 결국 연속 프레임)
- **확인 필요**: LeRobot이 delta_timestamps를 반올림해서 다른 프레임을 가져오는 경우, 이미지와 SAM/track 사이에 1프레임 어긋남 가능
- **검증 방법**: 학습 전에 adapter `__getitem__`에서 `sample["frame_index"]`나 `sample["index"]` 등을 출력해서 실제 매칭되는 프레임 인덱스 확인
- **수정 방법**: 만약 어긋나면, `frame_indices`를 LeRobot이 반환하는 실제 프레임 인덱스로 가져와야 함 (sample에 포함된 메타 정보 활용)

## 학습 후 할 일
- target 데이터로 eval (target 데이터는 학습에 안 씀, eval 전용)

