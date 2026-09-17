# RL2 paired eval / test 주의사항

## 동일 좌표의 개입 효과 검증

baseline과 개입 arm은 `(task, env_seed, policy_seed)`로 묶는다. 실행 순서, worker,
GPU 번호, arm 이름을 seed에 넣지 않는다. 체크포인트, 환경/코드 버전, 초기 상태,
action 전처리/후처리, horizon, SAFE 및 CP 설정도 동일하게 보존한다.

`paired_rng_v2`는 SHA256으로 task/env_seed/policy_seed/stream/step을 안정적으로
seed에 매핑한다. 각 episode 시작에 난수를 초기화하고, 각 action chunk의 기본
정책(`policy`)과 개입(`intervention`)은 독립된 RNG scope를 사용한다.
Python/NumPy/Torch CPU 및 보이는 CUDA RNG를 scope 종료 시 복원한다.
따라서 앞 episode의 조기 종료, 개입 횟수, 추가 noise sampling, worker 배치가
뒤 episode나 다른 stream의 noise에 영향을 주지 않는다. `pi0_policy.reset()`은
queue 초기화일 뿐 RNG reset이 아니다.

## 평가 전 및 결과 집계 게이트

- 먼저 CPU regression: `CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu .venvs/rl2_iid/bin/python -m pytest -q tests/test_rl2_paired_rng.py tests/test_rl2_stage2_loop.py`
- 실제 모델에서 동일 좌표 baseline/SAFE gate-off 소수 smoke를 수행한다. GPU 또는
  simulator 비결정성은 CPU 테스트로 검증되지 않는다.
- 각 arm의 episode 집합이 정확히 같고 중복/누락이 없어야 한다. 학습 reset과 평가
  reset은 분리한다. 여러 policy seed는 같은 reset의 반복이므로 CI는 reset 단위로 묶는다.
- 최초 개입 전 context/action/executed action이 동일해야 한다. 최초 개입 chunk에서도
  개입 전 context는 동일해야 한다. 무개입 episode는 길이/최종 성공 여부까지 동일해야 한다.
- `analyze.py`는 위 조건과 `rng_contract=paired_rng_v2`를 확인하고 불일치 시 중단한다.
  오차 허용치를 임의로 늘리거나 공통 완료 subset만 골라 통과시키지 않는다.
- RL2 noise는 동일하게 고정해도 개입 후 관측/궤적은 달라지는 것이 정상이다.
- 새 eval 출력 경로를 사용한다. request manifest에도 RNG 버전이 있어 구결과 자동 재사용은
  거절된다. 이전 평가와 새 평가를 섞지 않는다.

## 2026-09-17 발견한 구평가 문제

이전 코드는 25판 lane 시작 시 한 번만 seed를 설정했다. 개입의 추가 noise 소비와
rollout 길이 차이로 다음 episode의 noise가 달라졌다. 초기 context 불일치는 모두
앞 episode 개입 이후 발생했다. `outputs/rl2_iid_20260916/training/eval`은 이 구버전이며
paired rescue/destruction의 인과 근거로 사용하지 않는다. 원본은 보존한다.
학습된 SAFE/QAM 및 CP를 유지해 eval만 새 버전으로 재실행할 수 있다.

현재 사용자 자원 제한: kanu GPU 2/3/4만 사용. GPU당 모델 2개는 메모리/안정성 검증 시
허용되지만, evaluator가 해당 실행 모드를 지원하는지도 확인해야 한다.

## 축소 비교 실행 (2026-09-17)

`run_experiment.py --eval-arms qam_base qam_shaped --trials 12 --seeds 42 0 7`
은 각 arm 144판(4 task × 12 reset × 3 seed)을 실행한다. `--force-gate-off`는
모델 로딩 후 개입을 끄는 진단 전용이며 결과에 `force_gate_off=true`를 기록한다.
실제 성능 평가에 이 결과를 섞지 않는다. Vanilla는 smoke에서만 실행한다.

이번 실행 절차는 `outputs/rl2_paired_half_20260917/driver.py`에 고정했다.
Gate-off 24판의 전 action 일치 → natural gate 24판의 최초 개입 전 일치 →
기본/cluster reward QAM 288판 → reset 단위 paired bootstrap 순이다.
Smoke 실패 시 본 평가를 발사하지 않는다. 두 arm 비교에서는 어느 한쪽이 최초로
개입하기 전까지 context/action 일치를 확인하며, vanilla 없는 평가의 outcome 차이를
vanilla 대비 구제/파괴라고 부르지 않는다. GPU 2/3/4 중 비어 있고 lease 취득 가능한
장치만 사용하며, 각 단계 사이 추가 여유 장치를 확인한다.

## 환경 재사용 격리 및 해시 진단

난수 분리만으로 환경 내부 상태가 초기화되지는 않는다. `fresh_env_per_episode_v1`은
판마다 새 env 및 전처리 adapter를 만들고 종료 시 env.close()를 호출한다. 이전의
물리 장면을 reset해서 재사용하지 않는다. 실제 입력 noise를 명시적으로 생성해
select_action(noise=...)에 전달한다.

각 chunk에 raw observation/model observation/simulator state(지원 시), policy noise,
proposed action, 각 executed action의 SHA256을 저장한다. 최초 개입 전 입력 및 noise
해시를 먼저 비교하고, 일치할 때 activation/action을 비교한다. 관측 해시부터 다르면
환경/입력 경로를, 동일 입력·noise인데 activation이 다르면 모델 실행 경로를 조사한다.
새 진단/축소 평가 경로는 `outputs/rl2_paired_freshenv_20260917`이며 구 결과는 보존한다.

## 검증된 dev에서 전체 평가 복원

`run_full_paired_eval.py --output <새 경로> --reuse-root outputs/rl2_paired_freshenv_20260917 --run`
은 원래 네 arm을 각 300판으로 복원한다. 검증된 본 평가의 QAM 288판 및 natural smoke의
vanilla 8판을 재사용하고 904판만 새로 실행한다. Gate-off 결과와 구 환경 재사용 결과는
제외한다. Manifest에는 재사용 원본 파일·SHA256, 실행 dev 커밋·소스 해시·checkpoint 해시를
기록한다. 실행 소스가 dev 커밋과 다르거나 실행 도중 변경되면 다음 batch를 중단한다.
완료 시 1,200판의 정확한 좌표 집합 및 개입 전 일치를 검증한 후 전체 요약을 만든다.
별도 재현 smoke는 다시 실행하지 않는다.
