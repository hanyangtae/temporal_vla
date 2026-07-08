# Phase 분류기 학습용 데이터셋 (phase_cls_v1) 설계

- 날짜: 2026-07-08
- 상태: 설계 승인 대기
- 관련 연구 문맥: `docs/steering/14_pathway_phase_online_steering.md` — phase-matched steering의
  전제 조건인 **online phase 식별**을 supervised 분류로 검증하기 위한 데이터셋.

## 1. 목적

`outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts/`의 rollout pkl(540 에피소드)을
PyTorch 분류 학습에 바로 쓸 수 있는 벡터화된 형태로 변환한다.

- **학습 목표**: hidden state → 현재 phase 분류 (online phase 식별의 오프라인 검증).
- 라벨은 수집 시 시뮬레이터 privileged state로 판정된 값을 그대로 사용한다
  (`feature_phases`, `episode_success` — 별도 재판정 불필요).

## 2. 범위 / 비범위

**범위 (v1):**
- `raw_rollouts/` 540 에피소드만 변환 (성공 358 / 실패 182, 9개 cell).
- DiT 전 레이어 (7 layers × 4 denoise × 1536) + VL(2048) + action/proprio 보존.
- 성공/실패 에피소드 모두 포함, 학습 시 필터 가능하게.

**비범위:**
- `raw_ps*/xa*/xb*/gx*/cross*` 개입 실험 데이터 (pkl 스키마 동일 확인됨 —
  추후 같은 스크립트로 새 샤드 append, `source` 컬럼으로 구분).
- 분류기 모델/학습 코드 자체 (별도 작업).

## 3. 원본 데이터 사실 (확인 완료)

- pkl 1개 = 에피소드 1개. 시계열 길이 T = 정책 inference 호출 수
  (n_action_steps=5 → 실패 ep는 720/5=144 고정, 성공 ep는 가변, 예: 50).
- 스텝당: `hidden_states` (7,4,1536) fp16 torch.Tensor
  (capture_layers=[0,2,4,8,10,12,15], 축=[layer, denoise_step, feature_dim]),
  `vl_hidden_states` (2048,), `action_vectors` 행 (12,), `states` dict (proprio 16차원).
- 라벨: `feature_phases[t]` (event_state 스킴, sim 술어 기반),
  `episode_success` (파일명 `succ{0|1}`과 일치 — 표본 검증 완료).
- 관측 phase: reach-to-object / grasp / transport / place / insert-settle
  (+스킴상 terminal, wrong-grasp 가능 — `robocasa_event_labeler.py` 참조).

## 4. 저장 스키마 (승인됨 — 방식 A: memmap 샤드 + Parquet 인덱스)

위치: `outputs/eval/robocasa/groot_n15/phase_event_6p/datasets/phase_cls_v1/`

```
phase_cls_v1/
├── meta.json                 # 스키마 버전, phase vocab, 축 정의, 샤드별 행 수, 추출 설정
├── index.parquet             # 스텝당 1행
├── build_report.txt          # 빌드 요약 (분포표, 스킵 파일, 검증 결과)
├── splits/
│   ├── default.json          # episode 단위 (cell × success) 층화 70/15/15
│   └── holdout_potato.json   # ppcc_potato 전체를 test로 (unseen-cell 일반화 평가)
└── shards/                   # cell당 3개, np.load(mmap_mode='r')로 접근
    ├── raw_rollouts__ppcs_apple.dit.npy    # fp16 [n, 7, 4, 1536]
    ├── raw_rollouts__ppcs_apple.vl.npy     # fp16 [n, 2048]
    ├── raw_rollouts__ppcs_apple.extra.npy  # fp32 [n, 28] = action_vector(12) ⊕ proprio(16)
    └── ...
```

**index.parquet 컬럼** (스텝당 1행, 총 ~5만 행):

| 컬럼 | 타입 | 내용 |
|---|---|---|
| `shard`, `row` | str, int32 | memmap 주소 (샤드 파일명, 샤드 내 행) |
| `episode_id` | str | `raw_rollouts/ppcs_apple/ep12` — split 단위 |
| `t`, `T` | int16 | 에피소드 내 위치 / 길이 |
| `phase_id` | int8 | 라벨. vocab: 0=reach-to-object, 1=grasp, 2=transport, 3=place, 4=insert-settle, 5=terminal, 6=wrong-grasp |
| `episode_success` | bool | 성공/실패 필터 |
| `cell`, `task`, `seed`, `source` | str/int | 필터·층화·확장용 |
| `grasped`, `wrong_grasped` | bool | grasp/wrong_grasp timeline (보조 분석용). 원본 timeline은 길이 T+1이므로 `timeline[t]` (policy가 행동한 관측 시점 = 블록 시작)로 정렬해 T개만 사용 |

**포인트:**
- 에피소드를 샤드 내 연속 저장 → temporal window 확장 시 인덱스 산술만으로 지원.
- phase vocab은 meta.json에 고정 매핑. 추출 시 vocab 밖 라벨 관측 → 즉시 에러.
- 용량 ≈ 4.5GB (5만 스텝 × ~90KB).

## 5. 추출 스크립트

`scripts/safe/groot_n15/robocasa/dataset/build_phase_dataset.py`

- CLI: `--raw-root <...>/raw_rollouts --out <...>/datasets/phase_cls_v1 [--overwrite]`.
- 처리 단위 = cell 디렉토리. pkl을 episode_idx 순 정렬 → 스텝별 누적(cell당 ~500MB RAM)
  → npy 3개 저장 + index 행 생성.
- 에피소드별 정합성 검사 (실패 시 해당 에피소드 에러 처리):
  `len(hidden_states) == len(feature_phases) == len(vl_hidden_states) == len(states)
  == action_vectors.shape[0]`, 관측 phase ⊆ vocab,
  pkl `episode_success` == 파일명 succ 플래그.
- 손상/불일치 pkl은 스킵하고 build_report.txt에 기록.
- 출력 디렉토리 존재 시 `--overwrite` 없으면 중단.

## 6. Split 전략

- **분할 단위는 episode** (스텝 단위 분할은 인접 스텝 유사성 때문에 train→test 누수).
- `default.json`: (cell × episode_success) 층화 70/15/15, 고정 시드로 생성.
  JSON에 `{"seed": ..., "train": [...], "val": [...], "test": [...]}` 형태로
  시드와 episode_id 리스트를 함께 저장 (재현 가능).
- `holdout_potato.json`: ppcc_potato 전 에피소드를 test로.
- 기본 학습 필터 권장값: train은 `episode_success == True`(깨끗한 라벨),
  실패 ep는 평가·분석용 (요구사항: "모두 담고 필터 가능하게").

## 7. PyTorch 로더

`src/datasets/phase_cls_dataset.py` (기존 `src/datasets/` 컨벤션 준수)

```python
ds = PhaseClsDataset(
    root, split="train",
    pathway="dit",        # "dit" | "vl" | "both"
    layer=8, denoise=-1,  # denoise="mean" 지원
    success_only=True,
    in_memory=True,       # 필터+슬라이스 결과 RAM 상주 (단일 레이어 ≈ 150MB)
)
x, y = ds[i]              # x: float32 벡터, y: phase_id
```

- `in_memory=False` → memmap lazy-open (`__getitem__`에서 worker별 최초 1회 오픈,
  `num_workers>0` 안전).
- `ds.class_weights` — phase 불균형(reach 편중) 대응용 가중치 헬퍼.

## 8. 검증 (build 직후 자동)

1. 행 수 일치: index.parquet 총 행 == 샤드 npy 행 합 (meta.json 기록치와도 대조).
2. bit-exact 왕복: 무작위 30개 (episode, t) → 원본 pkl 재로드 → memmap 값과 동일성 비교.
3. phase 분포 히스토그램 + cell×success 에피소드 표 → build_report.txt.
4. pytest: vocab 매핑 고정 테스트, split 간 episode_id 교집합 공집합(무누수) 테스트.

## 9. 에러 처리 요약

| 상황 | 처리 |
|---|---|
| pkl 로드 실패 / 길이 불일치 / vocab 밖 phase | 해당 에피소드 스킵 + report 기록, 빌드 계속 |
| `episode_success` ≠ 파일명 succ | 에피소드 스킵 + report 기록 (라벨 신뢰 불가) |
| 출력 디렉토리 기존 존재 | `--overwrite` 없으면 즉시 중단 |
| 스킵 에피소드 > 5% | 빌드 실패로 종료 (원본 데이터 문제 의심) |

## 10. 성능 근거 (논의 완료)

- 읽기 전용 np.memmap은 락 없음, 페이지 캐시 worker 간 공유 → `num_workers>0` 안전.
- 샘플당 읽기 3KB(단일 layer/denoise)~86KB(전체 축), 데이터셋 4.5GB → 첫 epoch 후
  페이지 캐시 상주.
- 일반 학습 경로는 `in_memory=True`로 I/O 자체를 제거.
