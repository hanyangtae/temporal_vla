# GR00T 발사 하네스

## 적용 범위

2026-09-10 사용자 규칙: GR00T에만 적용. srv48·srv50 각각 **모든 세션 합산 GPU 1장 / GPU당 6 serve**. kanu **모든 세션 합산 GPU 3장 / GPU당 2 serve**. 별도 세션이 GPU를 추가로 받는 방식으로 한도를 늘릴 수 없다.

기계 정책은 `configs/harness/gpu_policy.json`, 사람용 설명은 `docs/05_gpu_server_rules.md`. worktree는 git common checkout의 정책을 읽는다. 원장은 kanu common checkout `outputs/gpu_leases` 단일 경로이며 shell lease와 Python 하네스가 같은 flock을 사용한다.

이번 구현은 **발사 admission + 추가 scene 수집 계약 검증**이다. 작업 큐의 자동 충원, detector/operator fit 산출물 감사, eval episode 중복 재사용·완료 집계는 아직 이 하네스에 통합하지 않았다. GPU 모델 수는 요청한 슬롯 수를 예약하며 실제 러너는 그 값으로 serve 수를 만든다.

## 연결된 진입점

- `scripts/safe/groot_n15/robocasa/collect/collect_grid.sh`
- `scripts/steer/online_gated/run_online_gated_eval.sh`

두 러너 모두 출력 생성 전에 `groot_launch_guard.sh`를 거친다. 기존 trap/cleanup은 유지한다. `gpu_lease.sh`도 공통 원장과 모든 세션의 GPU 장수 한도를 검사한다. 기존 shell lease에는 모델 필드가 없으면 GR00T로 간주한다. 다른 모델의 새 lease에는 `LEASE_MODEL=<실제 모델>`을 명시하며 GR00T 장수 상한을 적용하지 않는다(같은 GPU의 배타적 소유는 유지).

이미 실행 중인 작업은 중단/이관하지 않는다. **원격 서버 및 다른 worktree의 이전 코드에는 자동 적용되지 않는다.** 동일 코드를 배포하고 원격 coordinator 연결을 설정한 후부터 적용된다. 이전 러너·직접 SSH/Docker 명령을 차단하는 OS 권한 경계는 이번 범위에 없다.

## 발사

kanu에서는 `HARNESS_MACHINE=kanu`, `HARNESS_SESSION=<실제 소유 세션명>`을 지정하고 기존 러너를 실행한다. `SERVES_PER_GPU` 미지정 시 kanu=2, srv48/50=6. 명시한 더 작은 값도 허용한다. 세션명은 실행마다 바꾸지 않는다.

```bash
HARNESS_MACHINE=kanu HARNESS_SESSION=eval-main GPUS=6,7 \
  SLUGS=<대상> ARMS=<arms> <기존 실험 환경변수> \
  bash scripts/steer/online_gated/run_online_gated_eval.sh
```

위 `<...>`는 해당 실험의 실제 값으로 바꾼다. 하네스 도입 자체가 실험 조건을 정하거나 실행을 승인하지 않는다.

원격 러너에는 다음 두 변수가 추가로 필요하다.

```bash
export HARNESS_COORDINATOR_SSH='<원격 서버에서 kanu로 접속 가능한 SSH 별칭 또는 user@host>'
export HARNESS_COORDINATOR_SCRIPT='/home/dongkyu/pkt_ws/temporal_vla/scripts/utils/groot_harness.py'
export HARNESS_MACHINE=srv48  # worker1. worker2는 srv50
export HARNESS_SESSION=eval-main
```

연결값은 이 구현에서 추정하거나 새로 설정하지 않았다. coordinator 연결 실패 시 별도 로컬 원장으로 대체하지 않고 발사를 거부한다. 실제 hostname과 선언 machine이 다르면 거부한다.

현재 실행기는 **빈 GPU에서 묶음 러너를 시작**한다. 살아 있는 compute PID가 있으면 같은 사용자라도 추측해서 소유권을 가져오지 않는다. 따라서 한 GPU의 6개 arm은 가능한 한 한 번의 묶음 실행에 넣는다. 동시 별도 러너에 대한 슬롯 상한도 원장이 검사하지만 이미 기동된 GPU에 후속 러너를 추가하는 자동 충원 기능은 별도 구현이 필요하다.

## 추가 scene 수집 계약

수집 러너에는 `HARNESS_COLLECTION_CONTRACT`, `DONE_LIST`, `GRID_ROOT`를 명시한다. 계약 JSON 예:

```json
{
  "kind": "collect",
  "model": "groot",
  "machine": "kanu",
  "parent_plan": "/absolute/path/to/canonical/collection_plan.json",
  "parent_plan_sha256": "부모 파일 전체 sha256",
  "plan_sha256": "추가 수집 plan 파일 전체 sha256",
  "parent_grid_root": "/absolute/path/to/original/grid",
  "allowed_instructions": ["PPCC/bread", "PPCC/jug", "PPCC/marshmallow"],
  "excluded_instructions": ["CoffeeSetupMug", "PPCC/apple"],
  "jitter_count": 5,
  "noise_count": 10
}
```

`parent_plan` 상대 경로는 계약 파일 기준, 나머지 경로는 절대 경로 사용을 권장한다. SHA256은 `sha256sum <파일>`로 확인한다. 예시 allowed 목록은 PPCC 작업용이며 다른 작업은 승인된 실제 범위로 작성한다. coffee/apple 제외는 이번 실험 계약에만 해당하며 상시 규칙으로 하드코딩하지 않는다.

검사 내용:

- 부모·신규 plan digest, GR00T 모델, checkpoint/version/캡처 조건/정책 seed 동일.
- 기존 모든 instruction의 scene·seed·jitter prefix 그대로 보존. 신구 좌우 키를 임의 치환하지 않는다.
- 환경/언어 매핑과 target10 주방 목록 보존.
- 신규 scene에 layout/style 및 5개 jitter 재현 좌표 존재, 유한 오프셋, drawer의 5개 서로 다른 reset index.
- 10개 고유 noise seed. 실행 instruction은 allowed 범위 안이고 excluded 밖.
- `DONE_LIST`가 선택 instruction의 부모 좌표 전체를 포함. 기존 판 재수집 방지.
- 기존 grid와 분리된 output root, 수집 machine 일치.

이 검사는 **실제 drawer k-scan 성공이나 pull-side feasibility·원본 pkl 무결성을 증명하지 않는다.** 그 실측 검증은 수집 준비 절차에서 별도로 수행하고 frozen plan을 계약에 기록해야 한다. 현재 guard는 추가 scene 수집용 계약을 요구한다. 처음부터 새 데이터셋을 만드는 수집은 별도 계약 형식 지원 전까지 이 진입점에서 거부된다.

모델을 띄우지 않는 독립 검증:

```bash
python3 scripts/utils/groot_harness.py check-collection \
  --contract <contract.json> --plan <collection_plan.json> \
  --instructions 'PPCC/bread' --noise-limit 10 \
  --done-list <parent_done.txt> --grid-root <new_grid_root> --machine kanu
```

## 예약 수명과 장애

예약 receipt는 러너 하나가 소비한다. 다른 PID·다른 kind·다른 자원으로 재사용하면 거부한다. 다른 세션의 예약 해제도 거부한다. 예약은 시간만 지났다고 만료하지 않는다. 실행기가 죽어도 분리 실행된 모델 서버가 살아 있을 수 있기 때문이다.

정상 종료 시 GPU compute PID가 없음을 확인한 뒤 예약을 해제한다. 잔존 PID/NVML 오류 시 receipt를 로그에 남기고 예약을 유지한다. coordinator 장애나 강제 SIGKILL로 남은 예약은 소유자가 실제 프로세스 종료를 확인하고 해제한다.

```bash
python3 scripts/utils/groot_harness.py status
# 서버/수집/eval 프로세스가 종료됐음을 확인한 뒤에만:
python3 scripts/utils/groot_harness.py release --receipt <receipt> --session <owner>
```

receipt와 세션명은 운영상 소유권 확인 수단이며 보안 인증 수단이 아니다. 강제 접근통제가 필요하면 에이전트의 직접 GPU 실행 권한을 분리하고 실행 서비스에만 권한을 부여해야 한다. 이번 변경은 계정/SSH/Docker 권한을 변경하지 않는다.

## 검증

```bash
python3 -m unittest discover -s scripts/utils/tests -p test_groot_harness.py -v
bash -n scripts/utils/groot_launch_guard.sh scripts/utils/gpu_lease.sh
bash -n scripts/safe/groot_n15/robocasa/collect/collect_grid.sh
bash -n scripts/steer/online_gated/run_online_gated_eval.sh
```

테스트는 임시 원장·모의 NVML을 사용하며 실제 모델이나 수집 프로세스를 실행하지 않는다.
