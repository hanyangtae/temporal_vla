# 05 GPU 서버 운영·예약 규약 (GR00T N1.5 eval 단일 출처)

모든 Claude 세션이 kanu·srv48·srv50에 serve/eval을 올릴 때 여기만 본다.
발사 전 **반드시 `scripts/utils/gpu_lease.sh claim`** — 다른 세션이 잡고 있으면 기다리거나 사용자에게 묻는다.
(흩어져 있던 규칙 통합: CLAUDE.md 평가표준·robocasa-steer-eval 스킬·메모리 kanu/a100 규칙·핸드오프 §3. 충돌 시 이 문서가 이긴다.)

## 1. 머신별 규칙

| | **kanu** (로컬) | **srv48** (`AISem_48_junhyeong`, worker1) | **srv50** (`AISem_50_junhyeong`, worker2) |
|---|---|---|---|
| GPU | A4000 16GB × 8 | A100 80GB × 4 | A100 80GB × 4 |
| 사용 가능 GPU | **빈 GPU만**, 한 세션 최대 3장. 타인 프로세스(예: junhyeong `main.py` 436MiB 상주) 있으면 금지 | 빈 GPU만. 관례 **GPU2** | 빈 GPU만. 관례 **GPU1** (GPU0은 타인 상시 점유) |
| GPU당 serve | **2** (serve 상주 ~5.8GB) | **6** | **6** |
| serve 방식 | docker `lerobot` 컨테이너 (`docker exec -d`) | host conda `~/miniconda3/envs/lerobot_050_groot/bin/python` + `SERVE_PYTHONPATH=~/pkt_ws/temporal_vla/lerobot/src` | 좌동 |
| repo | `~/pkt_ws/temporal_vla` | `~/pkt_ws/temporal_vla` (git pull; NPZ·ckpt·번들은 tar 반입) | 좌동 |
| 포트 대역 | 8860~ | 8890~ | 8890~ |
| 머신 매칭 | 수집 머신에서 eval (v4 index `machine` 열) | 좌동 | 좌동 |

- **승준**(원격 CPU, `kimseungjun@166.104.146.37:11112`): GPU 없음. CPU 8코어 공유 → 스레드 cap 8, 무거운 job 동시 2 이하. 코드는 git만(scp 금지), 데이터는 tar 스트림. srv에서 직송 가능.
- 빈 GPU 판정 = `nvidia-smi --query-compute-apps=gpu_uuid,pid` 로 **프로세스 소유자까지** 확인. 메모리 잔량만 보고 판단 금지.
- 컨테이너 4일+ 가동 시 NVML 상실(`Failed to initialize NVML`) → `docker restart lerobot`. serve가 CPU로 뜨면 FlashAttention 에러로 위장 사망.
- 끝나면 **반드시 정리**: serve kill(포트로 식별) → `nvidia-smi`로 반납 확인 → lease release.

## 2. 세션 간 예약 (lease)

여러 세션이 동시에 발사하려 들 수 있다. 원장 = 이 PC의 `outputs/gpu_leases/`(전 세션 공통 FS). srv·승준 작업도 이 PC에서 발사하므로 같은 원장에 기록한다.

```bash
scripts/utils/gpu_lease.sh status                                  # 누가 뭘 잡고 있나
scripts/utils/gpu_lease.sh claim  kanu 4 "<세션명>" "<용도>" [ttl_h]   # 잡기 (기본 ttl 12h)
scripts/utils/gpu_lease.sh claim  srv50 1 "<세션명>" "<용도>"
scripts/utils/gpu_lease.sh wait   kanu 4 "<세션명>" "<용도>"            # 풀릴 때까지 대기 후 claim
scripts/utils/gpu_lease.sh release kanu 4 "<세션명>"                   # 반납 (본인만)
```

- claim은 원자적(mkdir). 이미 잡혀 있으면 실패 코드 3 + 소유자 출력 → **다른 세션 것이면 발사 금지**: `wait`로 기다리거나 사용자에게 보고.
- stale 자동 해제: 기록된 PID가 죽었거나 ttl 초과. (죽은 세션이 잡아둔 채 사라지는 것 방지)
- lease는 "GPU 1장" 단위. GPU당 serve 2/6는 같은 세션 안에서 나눠 쓴다(세션 둘이 한 GPU 공유 금지 — 포트·VRAM 충돌).
- 사용자 직접 발사분도 세션이 `status`에서 못 보므로, 발사 전 `nvidia-smi` 소유자 확인은 lease와 별개로 항상 한다.

## 3. 발사 체크리스트 (스킬 pre-flight와 같이)

1. `gpu_lease.sh status` → 겹치면 wait/보고
2. `nvidia-smi` 타인 프로세스 확인 (kanu는 특히)
3. 컨테이너/NVML 살아있나 (`docker exec lerobot nvidia-smi`)
4. 포트 충돌 (`pgrep -f "port 88"`)
5. claim → 발사 → 완료 감사(매니페스트 대비 행수) → serve 정리 → release

관련: 러너 `scripts/steer/online_gated/run_online_gated_eval.sh`(`ALLOW_BUSY_GPU`, `SERVES_PER_GPU`, `SERVE_MODE=host`), 스킬 `.claude/skills/robocasa-steer-eval/SKILL.md`(pre-flight 게이트).
