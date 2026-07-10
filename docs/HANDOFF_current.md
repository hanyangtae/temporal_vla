# HANDOFF — scene-seed 매트릭스 완료(판정 확정) + 재설계 라운드 실행 대기 (새 세션 진입점)

> 2026-07-10 갱신. **이 문서의 용법**: 새 세션은 ①§1–4로 현재 상태를 파악하고, ②§6 실행 플랜을
> 검토·보완한 뒤(§7 보완 포인트), ③§8 함정 목록을 숙지하고 실행에 착수한다.
> 설계 단일출처: [`docs/steering/17_steering_experiment_redesign.md`](steering/17_steering_experiment_redesign.md)(재설계 v2),
> [`docs/steering/18_apple_success_rejudge.md`](steering/18_apple_success_rejudge.md)(채점 재판정).
> ⚠️ 이 문서의 이전 버전(§1.1 "α=0.3 균일" 등)은 부정확 — git 이력에만 남기고 본 버전이 대체.

---

## 0. 한 줄

scene-seed 매트릭스(8 cell × ~23 arm, ~11,000 rollouts)가 양-머신(로컬 3GPU + A100 GPU2)
큐 시스템으로 완주됐고 판정은 **"고정 seed·no-SAE raw conceptor 접근 종결"**(교차-seed 재현 0).
단, 이 라운드는 **α 오배선·apple 채점 오류·fit 표본 부족**으로 과학표가 오염된 것이 확인되어,
문서 17/18 기반 **재설계 라운드(총 120 arm)** 를 실행하는 것이 다음 작업이다. 플랜은 §6에 확정돼
있고 사용자 결정 3건도 반영됨 — 새 세션은 보완 후 바로 실행하면 된다.

---

## 1. 매트릭스 결과 요약 (원판정 기준 — 동결 예정)

- 수치: `outputs/eval/robocasa/groot_n15/steer_eval/RESULTS_final_scene.json` (gxL 8 arm 열은
  재집계 필요 — §5-R1). 세부 판정 근거는 메모리 [[conceptor-steering-final-verdict]] 07-10 절.
- **교차-seed 재현성 = 0**: bread84 양성 arm(gatedps15 +8, gg60 +10)이 다른 bread 3 seed 전부에서
  부호 반전. 저SR bread 2 + apple 4 cell 전면 해악(최악 grand-gated15 8/60, base 55/60 대비).
- "L4 단일층 무해"는 α 문제로 C≈0 → **사실상 identity**(개선 아님). 그마저 bread84선 −16 파국.
- N/A 4 arm: s100084 ps15/ps30 (fit 창 실패 표본 부족 — 대조 fit 불성립, 구조적).
- **주의: 이 표는 원판정(채점 오류 포함) 기준.** corrected 재해석은 문서 18 §3–4 (오판정 10%,
  steering 해악 과대평가 — 단 gated 해악은 교정 후에도 유지).

## 2. 이 라운드를 무효화한 3가지 원인 (재설계 동기 — 문서 17 상단)

1. **α 오배선**: fit의 α 선택(overlap 밴드)은 매번 실행됐지만(0.1이 49%), serve가 NPZ **첫 키**를
   로드하는데 저장이 python set 순회라 우연에 좌우 — 실제 적용은 {0.1, 0.3} 혼재, 선택값 ≥1인
   23%는 조용히 0.3. → 재설계에선 STEER_ALPHA 명시 전달 필수. [[alpha-wiring-audit]]
2. **apple 채점 오류**: `PickPlaceCounterToStove` 성공 술어의 pan 중심거리 0.07 < pan 반경 0.23 —
   pan 안 안착의 10%를 실패로 오판. 재채점 도구 `scripts/safe/groot_n15/robocasa/eval/rejudge_success.py`
   검증 완료(replay fidelity 382/382, 머신 간 재현 OK). corrected 기준 = 중심거리 0.10.
3. **fit 표본 부족**: 고SR scene은 fit 창에 실패 2~6판 → 실패 클래스 대조 불성립.

## 3. 이번 세션(07-07~10)에서 구축·수정된 것 (재사용 자산)

### 인프라 (전부 검증·가동 실적 있음)
- **pq 양-머신 arm 큐**: `scripts/safe/groot_n15/robocasa/steer/pq/` — pq_lib(flock pop/requeue/
  ledger/MACHINE.txt 출처기록), `lane_runner_local_v2.sh`(⚠️ v1은 `${var:+X=y}` 셸 함정으로 layer
  인자 유실 — v2가 `env` 경유 수정본), `lane_runner_w2_v2.sh`(A100 산출물 **승준 직송** — 로컬은
  파이프만, 경량 manifest_w2.txt만 로컬 보관), `monitor_pq.sh`(이벤트 시 종료→보고),
  `rename_arms.sh`(arm 폴더 새 명명 일괄 mv, dry-run 지원, 러너 가동 중 자동 거부).
- **worker2(A100) serve**: `heldout_round_cell_host.sh` — 호스트 conda `lerobot_050_groot` +
  `PYTHONPATH=~/pkt_ws/temporal_vla/lerobot/src`(서브모듈 v0.5.1 강제). GPU2만, serve 6개=36GB.
  ckpt·robocasa 이미지·kitchen asset 11GB 이관 완료. [[a100-worker2-parallel-eval]]
- **hardware calibration**: 머신 효과 < n=60 검출한계 → cell 혼용 허용(각주 필수). trajectory는
  머신 간 발산 = 독립 재표본 (per-seed flip은 머신 내 한정).
- **재채점 파이프라인**: rejudge_success.py — 승준 20코어 5-샤드 + 로컬 병렬 실행 패턴 확립
  (에피소드당 fresh subprocess 필수 — gym.make 연속 생성 시 scene 오염).
- **fit**: `pq/apple_fits.sh` — lerobot 컨테이너 python(libero_bench env 소실 대응), staging
  심링크 **상대경로**(`ln -srf` — 절대경로는 컨테이너에서 깨짐), fit 무결성 게이트.
  ⚠️ 게이트 구멍: 클래스 표본 부족 시 빈 fit({} 2바이트)도 [done] 통과 — 재설계에서 보완 필요.
- **aggregate**: `aggregate_final_scene.py` — manifest_w2 폴백(승준 직송 arm), machine 열,
  xL 열 지원. (gxL 열은 미구현 — R1에서 추가)

### 문서·기록
- `docs/a100_offload_plan.md`(§6 hardware confound 처방), 문서 17·18(병렬 세션 작성).
- 메모리 신설/갱신: [[a100-worker2-parallel-eval]] [[alpha-wiring-audit]]
  [[conceptor-steering-final-verdict]](07-10 최종판정) [[feedback-verify-before-relay]].
- 커밋: `3a32240`(pq 시스템) `e62e702`(v2 러너·fits 수정) — 승준(sync/rung2-20260707 병합됨)·
  worker2(pq-sync-20260707)에 동기화. **GitHub origin push는 미완**(이 머신에 credential 없음 —
  사용자가 `git push origin exp/rung2-n15-phase-separation` 한 번 실행 필요).

## 4. 지금 돌아가고 있는 것 (새 세션 인수 대상)

- **gxL sweep 잔여**: gxL4/gxL812 16 arm 중 마지막 ~7 arm — 로컬 L5/L6(GPU5·6) + A100 W1/W2.
  감시: `pq/monitor_pq.sh` 재발사로 인수(이전 세션 watcher는 소멸). 완료 판정 = queue 0 + running 0.
- **apple 재채점**: 승준 5-샤드 + 로컬 1-샤드, 07-10 06:41 기준 76% —
  tsv는 `rejudge_matrix_apple/rejudge_m_s*.tsv`(승준 `workspace/temporal_vla` + 로컬 동일 경로).
- 완료 시 R1(구 라운드 마감) 실행 → §6.

## 5. 사용자 확정 결정 (2026-07-09~10 — 재설계에 이미 반영됨)

| 결정 | 내용 |
|---|---|
| fit 크기 | **{15, 30}만** (60 제거, COAST 표준=15). 총량 고정(A안): x fit15=240판 pool서 15판 등 |
| min-class | fit15 succ/fail **≥3**, fit30 **≥6**(비례). 층화 랜덤 + 고정 seed + episode manifest 기록 |
| 채점 기준 | **(07-10 확정 D1 — apple 한정)** eval/선택/보고 = corrected(0.10) 단일 채점(**제외 없음**) + 0.07 성공 수 병기 + discordant_rate 진단 전용. **fit 표본만** 0.07~0.10 걸친 애매 판 제외. bread 는 재채점 없음(원판정 그대로). ⚠️ 이전 기술("판정 갈리는 rollout 전 단계 표본 제외")은 의도를 잘못 압축한 것 — eval 제외는 post-treatment selection bias (Gate1, docs/collab/2026-07-10) |
| scene | **bread 4개 전부 재사용**(bread84·s300028·s300033·s400020 — 07-10 사용자 확정 "bread84 정도는 쓰자": bread는 게이트를 fit-가능 하한(fit30 실패≥6)으로 완화, 단 bread84(.78)·s300028(.83)은 천장효과 각주 필수, R3 재채점 후 실패<6이면 그 scene만 교체) + apple은 ppcs_apple 재사용 + **신규 3개 선발**(교정 후 실패 2~4판이라 대조 fit 원리적 불가 — 대안 없음, 후보 6개×60판≈2~3h) |
| scope 축 | ps(per-scene) / x(instruction-pool 공유 conceptor) / gx(전체-pool) — **전이 실험 없음** |
| layer/β | 후보 {quota top-1 single, multi 4-8-12} × β{0.1,0.3} — P2 선택 rollout에서 결정, tie는 보수(작은 β·적은 layer) |
| gx 우선순위 | 항상 뒤로(사다리 마지막) · layer sweep은 유의 판정 시에만 grand로 확장(이번 라운드에서 조건 발동 실적 있음) |
| 폴더 명명 | `ho_<perm|gated>_<per_scene|cross_scene|grand>_fit<N>[_L4|_L812]` — 새 라운드는 처음부터, 구 라운드는 rename_arms로 일괄 |

## 6. 재설계 라운드 실행 플랜 (검토·보완 후 실행)

상세 근거는 문서 17. 아래는 실행 순서 요약 (전체 5~6일, 6-lane):

- **R(~1일, 선행)**: ① gxL 완주 → gxL 열 추가 최종 집계+Notion(원판정 동결) → rename_arms --apply
  → 승준 아카이브(검증 후 로컬 정리). ② apple 재채점 완료 회수. ③ ~~bread 재채점 프로파일~~
  (**07-10 사용자 결정: bread 는 재채점하지 않음** — 원판정 그대로 채점·fit).
  ④ 배선: fit 스크립트에 {NPZ 키 명시순서, 층화 샘플링+manifest, corrected 라벨 입력, null-control
  (라벨 셔플), positive-only} 추가 / robocasa **fork**에 apple 술어 0.10 패치 커밋 / P2·P3 러너가
  fit meta의 선택 α를 **STEER_ALPHA로 명시 전달** / 빈-fit 게이트 보강.
  → **07-10 실행 세션에서 ①(집계·Notion)·②(불필요 판정 — 승준 단독 커버 확인)·④ 완료**
  (커밋 1093c9d·a6311c5·459c70f·daf532a·5615da9, Gate1 원장 docs/collab/2026-07-10-steering-redesign-gate1.md).
  추가 채택: fit/P2 는 scene 당 60판을 층화 **30/30 disjoint split**(fit-half/select-half)으로
  분리(P2 는 select-half 에서만 — in-sample rescue 재발 차단), P0 scene 선발은 게이트 통과 중
  **seed 순서**(SR 순위 금지), P2 선택은 하방-위험 규칙, serve preflight 로그 대조.
- **P0(2~3h)**: bread 4 scene은 기존 재사용(§5 scene 행 — R3 재채점으로 실패 하한만 재확인).
  apple만 신규 후보 6 seed(manifest `coast4_reused_remote/manifests/selected_instruction_seeds.tsv`
  751행) × 60판 수집 → corrected-일치 필터 → 게이트(succ≥15 & fail≥15) → 3개 선발 → 8 cell 확정.
- **P1(~3h, CPU)**: 8 scene × {ps,x,gx} × fit{15,30} refit (7 layer × α 5종) → cell/tier별 후보 JSON.
- **Pilot(0.5일)**: **bread84**로 P1→P2→P3 15 arm 절차 검증(문서17 원안 — bread84가 매트릭스에
  재편입되어 원안 복귀. 유산 수치(gatedps15 .917 등)와 선택값 sanity 대조 가능). 결과는 본실험 편입.
- **P2(0.5~1일)**: 후보 config × β × fit-seed(ep0–29) 30판 — corrected-일치 채점, granularity별
  선택(ps=scene/x=instruction/gx=전역), 수치는 selection_report.json에만(보고 금지).
- **P3(~2일)**: cell당 15 arm = base(재사용 3 cell skip) + **null 대조**(셔플 fit) + positive-only
  + {perm,gated}×{ps,x,gx}×fit{15,30} → **120 arm**, pq 큐 + 새 명명 + 직송 + MACHINE.txt.
- **집계**: aggregate_v2(corrected-일치 SR·가변 n·원판정 병기·null 대비 Δ·machine) →
  confound-audit → Notion·판정.

신규 파일 권장 위치: `scripts/safe/groot_n15/robocasa/steer/pq2/`
(p0_gate.sh / p1_screen.sh / p2_select.sh / build_p3_queue.sh / aggregate_v2.py).

## 7. 새 세션이 보완할 포인트 (실행 전 확인)

1. ~~문서 17 내부 불일치~~ → **07-10 해소**: 상단(fit{15,30}, 15 arm/cell=120)이 확정,
   P3 절 정정됨.
2. ~~bread 재채점 프로파일~~ → **07-10 사용자 결정: bread 재채점 안 함** (원판정 라벨 사용,
   scene 게이트도 원판정 실패 수 기준).
3. bread scene 게이트 완화(§5)의 통계 처리: bread84·s300028의 천장효과 각주 + R3 재채점 후
   실패 수 최종 확인(실패<6이면 그 scene만 후보 교체).
4. P2 후보 수(2~3 config × β 2 = 4~6 arm-equiv/granularity) 최종 확정.
5. positive-only의 적용 범위(전 cell vs 게이트-탈락 고SR scene 전용 — 문서17은 전 cell 1 arm).
6. x·gx fit의 scene 하한(x: scene당 ≥3, gx: scene당 ≥1 — 문서17 제안값) 확인.

## 8. 함정 목록 (이 세션에서 실제로 당한 것 — 위반 시 arm 증발·데이터 오염)

1. **실행 중 스크립트 수정 금지** — bash 이어읽기 어긋남(gated arm 4개 증발 사고). 신규 파일+스왑.
2. **pkill/pgrep 자기매칭** — 패턴 분할(`P='foo'"_bar"`) 또는 PID로.
3. **심링크는 상대경로**(`ln -srf`) — 호스트 절대경로는 컨테이너에서 깨져 빈 fit/TIMEOUT.
4. **`A=1 ${cond:+B=2} cmd` 함정** — 확장 결과는 env 지정이 아니라 명령어로 파싱됨(127). `env` 경유.
5. **디스크 풀 연쇄** — sed -i·printf가 큐 파일을 손상시킴. 모니터의 디스크 이벤트(<10GB) 준수,
   fit NPZ는 미사용 레이어 프루닝(dit_L4/8/12만 유지, 아카이브 후) 정책 유지.
6. **harness 백그라운드 작업이 세션 재개 시 재실행**될 수 있음(rsync 중복 실행 실증) — bg 작업은
   멱등하게 설계, 완료 판정은 산출물 기준.
7. **arm 재사용은 심링크로 하지 말 것** — 이번에 ps30/60↔6p 심링크가 정리 작업과 충돌해 소실 사고
   (승준 아카이브로 복원함). 재사용은 명시 복사 또는 집계 단에서 참조로.
8. serve NPZ 로더 첫 키 함정(§2) — STEER_ALPHA 항상 명시.
9. 빈 fit(클래스 0~2판)이 [done] 통과 — min-class 게이트를 fit 스크립트 안에서 강제할 것.
10. 에피소드당 fresh 프로세스(재채점 replay), gym.make 연속 생성 금지.
11. 공유 자원: 로컬 GPU0-3 동료·GPU 3개 상한 / worker2 GPU2만 / CPU cap OMP≤16 / 장시간 run
    setsid / 검증(rc=0+개수·용량) 전 삭제 금지.

## 9. 미결·사용자 대기 항목

- GitHub origin push (사용자 터미널에서 1회 — §3 끝).
- wrong-grasp 검출 false negative(폭 넓은 물체) threshold 보완 — 라벨러 개선 후보, 재설계 P0 전
  적용 여부 미정.
- worker2 `outputs/train`(26.6GB, 5월 N1.5 finetune ckpt 유일본 가능성) 삭제 보류 중.
- potato instruction(6/54로 게이트 탈락)의 처리 — 재설계는 bread/apple 2 instruction 유지.
- 구 라운드 fit 수집 원본(phase_event_6p/raw_rollouts, SAE 학습 재료 후보)의 로컬 보존 여부는
  SAE 착수 시 결정 (승준 아카이브는 완료 상태 유지).
