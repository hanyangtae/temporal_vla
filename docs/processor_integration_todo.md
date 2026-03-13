# Processor Pipeline 통합 검증 TODO

## 현재 상태

- 브랜치: `refactor/processor-pipeline` (dev에서 분기, PR 미생성)
- 커밋 3개 완료:
  1. `feat:` src/processor/ 모듈 추가 (base, types, factory, obs/action)
  2. `refactor:` calvin_eval.py, robocasa_vla_eval.py에 pipeline 적용
  3. `docs:` CLAUDE.md, README.md 업데이트

## 검증해야 할 것

### 1. DreamVLA × Calvin 통신 테스트 (최우선)

DreamVLA로 Calvin finetuning 후 Calvin 벤치에서 추론이 정상 동작하는지 확인.

**확인 포인트:**
- CalvinObsProcessor 출력(uint8 HWC)이 VLAClient → serve_dreamvla.py까지 정상 전달되는지
- serve_dreamvla.py 내부 전처리(uint8→float tensor, history 관리)가 기존과 동일하게 동작하는지
- CalvinActionProcessor의 gripper 이산화가 Calvin env에서 assert 안 걸리는지
- action 차원(7D)이 모델 출력과 일치하는지
- act_step(multi-step prediction) 버퍼링이 정상 동작하는지

**실행 방법:**
```bash
# 1. DreamVLA 서버 실행
docker compose run --rm dreamvla \
  python /temporal_vla/scripts/serve_dreamvla.py \
    --checkpoint <finetuned_checkpoint> --precision bf16

# 2. Calvin 평가 (짧게 1 sequence만)
docker compose run --rm calvin \
  python /temporal_vla/scripts/calvin_eval.py \
    --dataset-path /temporal_vla/data/calvin/task_ABC_D \
    --server-url http://localhost:8200 \
    --num-sequences 1 \
    --num-videos 1 --video-dir /temporal_vla/outputs/processor_test
```

### 2. RoboCasa 통신 테스트

```bash
# DreamVLA × RoboCasa
docker compose exec robocasa python /temporal_vla/scripts/robocasa_vla_eval.py \
  --task TurnOnMicrowave --vla-server http://localhost:8200 \
  --num-rollouts 1
```

**확인 포인트:**
- RoboCasaObsProcessor의 키 리매핑이 기존 하드코딩과 동일한 결과를 내는지
- RoboCasaActionProcessor의 7D→12D 매핑이 정확한지
- state(32-dim)가 정상 전달되는지

### 3. 통과 후

- `dev`로 PR 생성
- 다른 모델(UP-VLA, X-VLA)도 순차 검증
