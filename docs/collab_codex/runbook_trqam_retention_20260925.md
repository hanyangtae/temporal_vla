# 성공 rollout 기반 QAM/TRQAM 이식·학습

## 실험 계약

- 출발: 제공된 Bridge QAM 500k. deployed `actor_fast`, critic, target critic 전부 복원.
- 기준 정책: 원래 actor_fast를 slow/target slow 슬롯에 복사해 고정. BC prior 갱신 없음.
- 이는 upstream TRQAM의 prior BC/EMA 업데이트와 다른 **고정 기준 정책 fine-tuning 변형**이다.
- 성공 train 432 episode / 3566 chunk. π0 수집 성공이며 QAM 성공 rollout이라고 해석하지 않는다.
- 원래 전체 train 데이터로 fit한 cluster potential 고정. 성공만으로 cluster 재fit 금지.
- 기존 QAM base reward(-1, 성공 마지막 3 step 0), gamma .99, shaping .1 그대로.
- partial chunk 실제 길이 할인과 mask 유지. 샘플은 chunk 균등이며 길이 보정 없음.
- 두 arm: fixed-reference QAM vs fixed-reference TRQAM. frozen original은 무학습 비교 기준.
- 동일 batch256/lr3e-4, reset optimizer, 복원 critic(초기화·warmup 없음), seeds42/0/7, 각50k update.
- TRQAM KL budget1, lambda init1, scale3, dual lr.01, EMA.1, clip2, lambda [.01,100].
- QAM inv_temp.1은 원래 값. TRQAM은 adjoint의 temperature를 제거하고 sigma로 lambda를 반영.
- loss 크기/gradient 차이는 두 알고리즘 자체의 차이다. 공식 budget을 SIMPLER 최적값이라고 주장하지 않는다.
- raw/max/p95 path KL, clipped/EMA KL, lambda, Q/TD, 원 정책 action 변화량을 함께 기록.
- 평균 path KL은 per-state 보존 보장이나 VLA 혼합 정책 KL이 아니다.

## 실행 및 검증

`train.py --smoke`는 실제 GPU에서 strict checkpoint load, step0 action·10 denoise vector 일치,
critic 복원, 3 update finite/actor 변화/reference 고정, 기존 QAM loader 재로딩을 검증한다.
환경과 checkpoint는 `requirements.txt`, flags.json 및 SHA로 남긴다.
CPU 단위 테스트: `JAX_PLATFORMS=cpu PYTHONPATH=RL2-VLA/third_party/qam <python> scripts/rl2_vla/trqam_retention/test_agent.py`.

원격 전용 worktree와 venv를 사용한다. 공유 환경 설치는 사전 허가 필요.
코드는 git으로, 체크포인트와 유도 cache만 remote_compute.sh push-data로 전달한다.
48 GPU3은 2026-09-25 점검에서 빈 GPU였으나 실제 발사 직전 재확인해야 한다.
50은 당시 전 GPU 타 작업 점유. 타인 job 종료·GPU 공유 금지. kanu 실험 금지.
GPU lease는 kanu 공통 원장에만 만들고 원격 종료까지 coordinator가 유지·반납한다.
GR00T GPU당6모델 규칙을 이 학습에 적용하지 않는다.

환경 변수 예시:
```
TRAIN_PYTHON=<isolated-venv>/bin/python
ASSET_ROOT=/home/junhyeong/pkt_ws/temporal_vla
RUN_ROOT=<worktree>/outputs/trqam_success_retention_20260925
CUDA_VISIBLE_DEVICES=3
bash scripts/rl2_vla/trqam_retention/run_arm.sh qam
```
`run_arm.sh trqam`은 같은 seed 순서로 비교군을 학습한다. 재실행 시 기존 output이 있으면 실패하며 덮어쓰지 않는다.
중간 checkpoint:1/10/100/1000/5000/10000/25000/50000. dual 상태는 별도 JSON.
flags.json은 모든 입력 SHA·episode 목록·하이퍼파라미터를 기록한다.

## 평가 범위

학습 종료는 성공률 유지 검증을 뜻하지 않는다. 이후 기존 paired RNG/fresh env 계약과 동일 SAFE/CP/VLA mixing으로 original/QAM/TRQAM을 평가한다.
이미 본 300판은 진단용이고 최종 확인에는 새 heldout reset을 쓴다.
이 성공-only pilot은 retention/행동 변화 확인용이며 실패 구제 성능의 결론은 낼 수 없다.
정책 보존 허용 폭과 평가 표본량은 실제 정책 평가 전에 확정해야 한다.

## 출처

- https://github.com/yonghdong/trqam commit `6f74d36baf552565fcd373198be25541db10abe3`
- https://arxiv.org/html/2605.27079v1
- 기존 QAM 구현 `RL2-VLA/third_party/qam/agents/qam.py`
