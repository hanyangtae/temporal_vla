# pq3 Phase C 핸드오프 (A·B 완료 → C 수집)

작성: 2026-07-16. 설계 단일 출처는 여전히 **`~/.claude/plans/dynamic-riding-aurora.md` (v9)** +
`docs/steering/19_pq3_execution_handoff.md`. 이 문서는 Phase A/B 실측 결과·동결 상수·사용자
결정·C 이후 절차만 담는다. 충돌 시 계획서 우선. 실행 런북:
`scripts/safe/groot_n15/robocasa/steer/pq3/README.md`.

## 0. 명칭 규약 (사용자 지정)

서버는 **승준(seungjun)** = kimseungjun@166.104.146.37:11112 / **a100 50** =
junhyeong@166.104.35.50 / **a100 48** = junhyeong@166.104.35.48 / **로컬** 로 부른다
(w1/w2/w3 호칭 금지 — 스크립트 파일명 `*_w2.sh` 는 관례상 유지, a100 계열 공용).

## 1. 현재 위치 (2026-07-16)

- 브랜치 `exp/pq3-coast-align` (분기점 ea61d24). **Phase A 완료 + Gate A 9항목 통과(lerobot
  컨테이너) + Codex Gate 2 3왕복 종결(지적 19→13→5건 전건 반영) + Phase B 스모크 완료.**
- 원장: `docs/collab/2026-07-13-pq3-gate1.md`, `docs/collab/2026-07-15-pq3-gate2.md`.
- push: 승준·a100 50 remote 완료. **origin(GitHub)은 자격증명 부재 — 사용자가 직접**
  (`git push -u origin exp/pq3-coast-align`).
- manifest(plan 단계)는 3 cell 생성 완료:
  `outputs/eval/robocasa/groot_n15/steer_eval_pq3/manifests/{pq3_drawer_left,pq3_drawer_right,pq3_ppcc_bread}/collect_plan.tsv`

## 2. Phase B 결과 요약 (전항 통과)

- **S1 T 실측 = 49 확정** (1 state + 32 future + 16 action). record = [L=7, K=4, T=49,
  D=1536] fp16, capture_token_mode=all_token_full, VL post-SA full-token **[T_vl=813, 2048]**,
  env-step GT mismatch=0.
- S2 유닛(Gate A) / **S2b β=0 sham 10판 = base 와 성공·step 완전 일치** / S2c gated 스위칭
  (등록 phase만 gated=True, 미등록은 identity 기록, phase×step 노름 영구화, 언더파이어 무오류)
  / S3 per-step NPZ 왕복 / **S4 natural-reset 재현성**(동일 seed·noise → 전 필드 동일) /
  S6 판정 특이 0/10 / S7 격리 / S8 push 완료.
- **S5 drawer 7-phase 분포** (base 10판, tail seed): 성공(3ep) reach 47%·grasp-handle 27%·
  pull 26% / 실패(7ep) reach 58%·disengage 35%·grasp-handle 6%·**pull 0.1%(1 record)**.
  → **pull phase 는 실패 클래스 희소.** 사용자 제시 선택지(Phase D dwell 표 보고 결정):
  ① 해당 phase 는 **success-only steer**(C_success 사용) ② **none**(steer 안 함 = identity)
  ③ **인접 phase 병합**. gated 성립 게이트(`pq3_gated_gate.py`)와 함께 사용자 게이트에서 결정.
- S9 host canary: 로컬 기준선 10판만 확보 — **3-host canary 는 E 직전** (당시 a100 50 GPU
  만석, a100 48 은 robocasa 컨테이너 GPU 상실 상태였음 — 사용 전 컨테이너 재시작 협의 필요).
- base SR 참고치: drawer_left tail 10 seed 에서 3/10.
- **용량 실측**: timeout(720step) 실패 episode 의 full-token pkl = **1.09GB/ep**
  (7.55MB/record × 144 records, VL 3.3MB/record 포함). 성공 episode 는 짧아 수십 MB.
  **사용자 결정: VL full-token 수집 유지** (최악 전부실패 기준 cell당 ~16GB, 5 cell ~80GB
  승준행 — 승준 HDD 여유 수집 전 확인).

## 3. 동결 상수 (구현 반영 완료 — 변경 시 Gate 재통과 필요)

| 항목 | 값 | 위치 |
|---|---|---|
| noise 시리즈 | collect = 500000 + tsv_cell_index×100000 + i×1000 / eval = 3000000 + tsv_cell_index×100000 + ep×1000 (arm 공유, 클라이언트 intra-episode +1) | `pq3/make_pq3_manifests.py` |
| α grid | Table14 GR00T {0.1,0.3,0.5,0.8,1,1.5,2,3,5,10} — fit 에 `--alphas table14` **명시** (CLI 기본은 pq2 legacy) | `fit_phase_conceptor_n15.py` |
| quota floor | 0.01 **명시** (`--quota-floor 0.01`, 기본 0=미적용) | 〃 |
| Stage2 밴드 | overlap [0.85, 0.95] | 〃 |
| Stage1 | α₀=10, per-step 동일 객체 quota, epeq(episode-equal) 순위 병기 — **사용자 게이트** | 〃 `--stage1-quota-sweep` |
| p0 게이트 | 수집 15판 ∧ succ≥3 ∧ fail≥3 (env 원판정), backfill 순서 S15.. | `pq3/p0_gate_pq3.py` |
| **seed 풀 제약** | drawer L/R·bread 는 소스 tsv 가 cell당 **50개** → fit15+backfill+unseen15 ≤ 50 ⇒ **backfill B ≤ 20** (fit30 까지 하려면 B≤5). 초과 시 `select_instruction_seeds.py --exclude-selected-seeds` 로 풀 확장 후 plan 재생성 | §2 참조 |
| eval 30 | seen 15 = fit 실사용 seed 첫 15(새 noise) + unseen 15(교집합 0) — freeze 가 동결(sha) | `make_pq3_manifests.py freeze` |
| β sweep | fit seed 재사용, β{0.1,0.3}, 하방 = wins > base−2 생존(−2판이면 탈락), 동률→0.1, red flag = 양 β 모두 −4판(rc6), 생존 0 = rc7 사용자 게이트 | `pq3/beta_decide.py` |
| 판정 | {H1,H2,H3}×{drawer,ppcc} 6 Holm(α=.05, 단측 exact McNemar), null 관문 ±4판, 비재현 CI = paired bootstrap(seed 20260715) 상한 < +0.16, gated N/A 는 gate report 자동 파생 | `pq3/pq3_decision.py`(동결, sha 기록) |
| serve/수집 | CAP 7층 0,2,4,8,10,12,15 · CHUNK_LEN 16 · n-action-steps 5 · max-episode-steps 720 · 캡처 OFF eval = `--no-features`(skip_features chunk 경로) | `pq3/pq3_cell_runner.sh` 등 |

## 4. Phase C 절차 (새 세션에서 수행 — 사용자 지시 2026-07-16)

**중단 시점 스냅샷** (이전 세션이 C 를 잠깐 기동했다 사용자 지시로 전부 중지·프로세스 정리):
- 부분 수집: drawer_left 0판 / drawer_right 1판(csv+**로컬 pkl 잔존, 미직송**) /
  ppcc_bread 1판(직송 완료 — SHIPPED.tsv 1행, 직송 파이프라인 실동작 확인됨).
- **재개는 그대로 러너 재실행이면 됨** (resume-safe: csv 스템 존재 시 수집 스킵, 잔존
  로컬 pkl 은 SHIP 루프가 재직송). C0 스캔도 `--resume` 사이드카가 있어 이어서 스캔됨.
- GPU·serve·수집 프로세스는 전부 정리됨 (전 GPU 유휴).

1. **fit15 수집** (로컬 GPU 0/1/2, cell당 GPU 1개·serve 2개, **SHIP=1 승준 직송**):
   `pq3/README.md` §C 명령 그대로. SHIPPED.tsv(size+sha ledger)·5MB 상식 체크·원격 실물
   대조가 내장돼 있음(유실 사건 표준). natural reset 강제(ep-meta replay 금지).
   **직송 목적지(실측 확정)**: `SJ_ROOT='~/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts'`
   — 승준 `~/datasets` 가 HDD(sda2, 283G 여유), `~/workspace` 는 **NVMe 라 금지**
   (스크립트 기본값이 workspace 이므로 SJ_ROOT 오버라이드 필수).
2. **C0** (병렬): `bash pq3/pq3_c0_scan.sh` — ppcc 신규 2종(빈도 기준, potato·bread 제외,
   60+ seed). 완료 후 ① `configs/robocasa/pq3_ppcc_new_cells.tsv` 생성됨 ② `pq/pq_lib.sh`
   CELLS 에 objA/objB 행 기입 ③ 신규 tsv 로 plan 생성.
3. cell별 `p0_gate_pq3.py` → rc2 면 BACKFILL_EPLIST 로 추가 수집(B≤20 주의), rc3 = cell 탈락
   (3연속 탈락 → **Gate C 4-cell fallback 사용자 게이트**).
4. 게이트 통과 시 `make_pq3_manifests.py freeze` (--pkl-prefix 는 승준 절대경로:
   `~/workspace/temporal_vla/outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts/<TASK>/<cell>`)
   → `pool` 로 task-pooled fit manifest 2개(OpenDrawer, PPCC).
5. 이후 Phase D (승준, anaconda python, 스레드 cap): Stage1 sweep → **사용자 게이트** →
   본 fit(per-step NPZ) → gated 성립 게이트(--enforce, pull-fail 선택지 결정 포함) →
   β sweep → hash 동결. Phase E: queue+lane(로컬 0/1/2 + a100 50 + a100 48, cell-블록 host).
   Phase F: `aggregate_pq3.py` (+confound-audit 필수).

## 5. 남은 사용자 게이트 / 미결

- 사용자 게이트: **Stage1 layer 결과**(D), **Gate C 4-cell fallback**(조건부),
  β 생존 0(rc7)·red flag(rc6) 시.
- 미결: origin push(사용자 직접) · S9 3-host canary(E 직전) · a100 48 robocasa 컨테이너
  GPU 상실(사용 전 재시작 협의) · a100 50 GPU 점유 상태 확인 후 lane 투입.
- pq2 fit raw 유실 사건(07-16 공지) 관련: pq3 는 심링크 서브셋 없음(manifest 직접 참조),
  직송 3중 검증 내장. **컨테이너 생성/재시작 금지**(문자열 검사로 갈음) — 단 GPU 상실 시
  재시작은 사용자 승인으로(로컬은 07-16 승인 하에 lerobot/robocasa 재시작 완료).

## 6. 함정 (이 세션 실증 — 19번 문서 §4 에 추가)

- `docker exec` 에 heredoc/stdin 쓸 때 **`-i` 필수** (없으면 무음 무실행).
- serve 의 module logger INFO 는 로그 파일에 안 남는다 — 러너 대조는 print 기반
  `[steer-preflight]`/`[steer-registered]`/`[serve-boot]`/`[steer-norms]` 라인만 사용.
- `codex review` 서브커맨드는 이 호스트 `use_legacy_landlock` 과 비호환(행) — Gate 리뷰는
  ask lane (`codex_ask.sh ask/resume`). instructions 는 --base/--commit 과 동시 사용 불가.
- 장기 실행 컨테이너의 GPU 상실(NVML Unknown Error) = cgroup/드라이버 리로드 증상 —
  컨테이너 내 `torch.cuda.is_available()` 로 확인, 재시작은 사용자 승인.
- `${VAR:+X=v} cmd` 식 조건부 env 주입은 rc=127 (확장 후 대입어 재분류 안 됨) — env 배열 사용.
- groot `select_action` 은 16-큐 팝 — 평가·수집 모두 chunk 경로(predict_action_chunk /
  skip_features)만 사용할 것.
