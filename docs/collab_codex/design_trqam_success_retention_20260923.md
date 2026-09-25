# TRQAM을 이용한 SIMPLER QAM 성공 행동 보존 실험

2026-09-23. 논문·공식 소스·현재 체크포인트 flags 검토. 이 문서는 적용 설계이며,
TRQAM 이식 또는 학습/eval을 실행했다는 뜻이 아니다.

## 적용 가능성과 차이

공식 TRQAM은 JAX/Flax flow actor 및 critic을 사용한다. 로컬 checkpoint는 context 1024,
4×7 action chunk, flow_steps=10, actor/critic width 512×4, residual=false다.
따라서 구조적으로 이식 가능성이 있으나 실제 parameter-tree strict load 및 step-0
velocity/action 일치를 먼저 확인해야 한다. 공식 benchmark는 OGBench/Robomimic이며
SIMPLER 즉시 호환을 입증하지 않는다.

공식 main.py의 pretrained_actor_path는 modules_actor_slow를 우선 선택해 fast/slow에
복사한다. 로컬 RL2 실제 추론은 residual=false에서 actor_fast를 사용한다. 로더를 그대로
쓰면 학습 없이도 정책이 달라진다. 여기서는 원 checkpoint actor_fast로 trainable actor와
고정 reference를 초기화한다. reference와 그 target은 업데이트하지 않는다. 공식 agent의
flow-matching prior 업데이트 및 target_actor_slow EMA를 그대로 복사하지 않는다.
이는 frozen deployed-QAM reference에 맞춘 TRQAM adaptation이며 공식 recipe 재현과 구분한다.

critic/target_critic은 호환되는 기존 가중치를 로드한다. 소량 성공 데이터에서 무조건
랜덤 초기화하지 않는다. optimizer 새 상태를 사용하는 경우 QAM/TRQAM 양쪽 동일하게
명시한다. reward/discount/action normalization 및 checkpoint의 Q scale 호환성을 검증하고
critic-only 적응 중에는 actor와 reference를 고정한다. 적응 길이는 validation으로 결정하며
critic을 유지한다고 자동으로 올바른 critic이 되는 것은 아니다.

공식 path-KL 가중치, horizon normalization, sigma/lambda 적용, projected dual update를
기준으로 포팅한다. 임의 velocity L2를 TRQAM이라 부르지 않는다. raw KL과 EMA/clipped KL을
함께 저장한다. 배치 평균 KL 제약은 개별 실패 state에서의 보존/복구 보장이 아니다.
또 training SDE의 path KL을 RL2의 VLA+QAM 합성 정책 KL로 동일시하지 않는다.

## 데이터와 reward 고정

`configs/experiments/trqam_success_retention_20260923/design.json`에 train-success episode
432개를 고정했다: eggplant163, spoon87, stack116, carrot66. 기존 π0 성공 rollout이며,
원본 QAM의 성공 궤적이라는 뜻이 아니다. task/reset으로 분리된 기존 split을 유지한다.
확인된 평가 성공판을 train으로 옮기지 않는다.

첫 실험에서는 기존 base reward 및 기존 train-only success/failure cluster potential을
그대로 사용하고 actor/critic 학습 transition만 성공 episode로 제한한다. 성공-only로
성공/실패 대조 potential을 다시 fit하면 실패 통계가 없어져 다른 실험이 된다.
향후 새로운 dense reward나 명시적 phase condition은 별도 축으로 추가한다.

## 단계별 검증

1. checkpoint 이식: strict load, 원본 actor_fast 대비 step-0 velocity 및 동일-noise action
   일치. step-0 identity는 통합 correctness 검사이며 과거 환경 재현 실험 반복이 아니다.
2. CPU/짧은 학습: reference hash 불변, nonzero actor 업데이트, finite critic/adjoint,
   lambda update 방향, clipping 포화, 실제 horizon=4 KL normalization 테스트.
3. 동일 성공 데이터·reward·minibatch·update 예산으로 fixed-reference QAM과 TRQAM 비교.
   frozen original QAM은 학습 없는 기준. trust-region budget은 별도 validation에서 선택.
   여러 training seed와 중간 checkpoint를 저장한다. 처음부터 50k만 학습 후 최종만 평가하지 않는다.
4. teacher-forced 진단: held-out activation 및 같은 noise에서 reference 대비 velocity,
   최종 action, gripper sign, path-KL 분포를 기록. train-success/heldout-success/failure-stagnation
   상태를 분리하고 마지막 항목은 학습하지 않은 진단용으로 표시한다.
5. 실제 eval: 기존 SAFE/CP/mix weight/N=1 고정. paired noise/fresh-env/hash 검증 유지.
   기준은 원본 QAM 성공판 보존, 총 SR, 파괴율, 구제율, 각각 alarm 모수다.
   SAFE가 개입하지 않은 판만으로 보존을 주장하지 않는다. 게이트가 활성화된 성공판을 별도 집계.

## 판정 범위

성공-only 실험은 finetuning retention/stability를 본다. 실패·정체에서 필요한 행동 변화가
충분한지 또는 복구 정책을 학습했는지는 입증하지 않는다. 정책이 전혀 안 바뀌어 보존되는
trivial 결과를 배제하려면 업데이트/velocity 변화와 KL 반응을 확인한다.
보존은 단순히 p>0.05라는 이유로 인정하지 않는다. 비열등성 허용 폭과 성공판 파괴 허용치를
본 평가 전 결정하고, paired 신뢰구간이 넓으면 '표본 부족'으로 기록한다.

보존 기준을 통과한 이후에만 실패·정체/복구 데이터를 포함한 동일 데이터·reward·예산
QAM-vs-TRQAM 비교로 확장한다. 성공-only를 영구 학습 데이터 정책으로 일반화하지 않는다.

## 근거

- 논문: https://arxiv.org/html/2605.27079v1
- 공식 실행 recipe: https://github.com/yonghdong/trqam
- 공식 loader: https://github.com/yonghdong/trqam/blob/main/main.py (pretrained_actor_path)
- 공식 알고리즘: https://github.com/yonghdong/trqam/blob/main/agents/trqam.py
- 로컬 추론: RL2-VLA/RL2_CoVer_VLA/simpler/rl2_utils.py QAMInference
- 기존 학습: scripts/rl2_vla/stage2_cluster_reward/train_qam.py
