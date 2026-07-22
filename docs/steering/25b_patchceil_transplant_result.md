# patchceil 결과 — donor activation transplant는 실패를 뒤집지 못함 (primary null)

2026-07-16. 계획 v2.1(`~/.claude/plans/pq3-wise-mist.md`), 사전등록
`scripts/safe/groot_n15/robocasa/steer/patchceil/PROTOCOL.md`(실행 전 동결),
Gate 1 원장 `docs/collab/2026-07-16-patching-transplant-gate1.md`.

## 질문과 설계 (1문단)

실패가 확정된 episode를 같은 (scenario_seed, inference_seed)로 재실행하면서, target의
first_grasp record부터 **같은 cell 성공 episode(donor)의 DiT L15 full-token activation을
denoise-step 정렬로 통째 대입**하면 실패가 성공으로 뒤집히는가. 대상 = ppcc_bread_s300033
(실패 40) + s400020 (실패 37) = 77판 전량, donor 4개/cell round-robin, 창 = donor 고갈까지
(고갈 후 자유 진행). oracle-assisted 존재 증명이며 배포 방법 아님.

## 결과 (n=77 targets × 5 arm)

| arm | s300033 (n=40) | s400020 (n=37) | 합산 |
|---|---|---|---|
| nopatch (무개입 재실행) | 0 | 0 | **0/77** (결정론 재현 ✓) |
| **donor (본 개입)** | **0** | **2** | **2/77 (2.6%)** |
| placebo-fail | 0 | 0 | 0/77 |
| donor-shuffle | 0 | 1 | 1/77 |
| **direct action-replay (대조)** | **4** | **8** | **12/77 (15.6%)** |

- paired exact McNemar: donor vs placebo b=2,c=0 p=0.25 / donor vs shuffle b=2,c=1 p=0.50
  → **primary p_final = 0.50 (α=0.05) — 비유의. PROTOCOL hierarchical gate에 따라
  exploratory(t0 스윕·early/mid layer·짧은 창) 미실행** (사용자 지시로 대기).
- donor별 분해: 8개 donor 중 s400020 ep34·ep36만 각 1건 — 특정 donor 효과 아님.

## 무결성 (anchors, 전부 통과)

- **A1 결정론**: pass B 재수집 16판 succ·actions 승준 원본과 bitwise 일치 + nopatch
  77/77 실패 재현 (판정 제외 0건).
- **A2 배선(cross-scene action-equivalence)**: 상대 cell env + donor seed + 전창 L15
  이식 → emitted actions == donor 저장 actions (양방향 max|Δ|=0.0) — 이식이 발화·정렬·
  행동지배 전부 정확함을 실측.
- **A3 sham**: 자기 activation 이식 4판 = baseline과 완전 일치, 여전히 실패.
- status 사이드카(발화 창) 검증 무효 rollout 0건.

## 해석 — 왜 action-replay(16%)보다도 못한가

A2가 메커니즘을 그대로 보여준다: L15 전창 대입 시 action = ε_target + Δt·Σv_donor —
즉 **L15 이식은 "초기 noise 오프셋이 낀 open-loop donor 재생"**이다. 관측을 계속 보는
closed-loop 이점은 파괴되고(L15 뒤에는 norm+proj뿐), 깨끗한 action 재생(16%)에 ε 오프셋
해악까지 얹혀 2.6%로 떨어졌다. 부가 발견: **개입으로 뒤집을 수 있는 창 자체는 존재**한다
(action-replay 16%, 특히 grasp-정체형 s400020에서 22%) — 실패의 다수는 "정책이 모르는"
게 아니라 grasp 진입 시점에 상태가 민감하게 어긋난 유형.

## confound-audit

| # | Gate | 판정 | 근거 |
|---|---|---|---|
| 1 | Length | N/A | 지표가 per-episode 성공 플립(길이 feature 아님). 지속 창의 donor-길이 종속은 Gate 1 규약대로 명시, null이라 양성 오염 여지 없음 |
| 2 | Task identity | N/A | 단일 task·within-cell 비교만 |
| 3 | Instruction | pass | canonical instruction 단일 고정 |
| 4 | In-sample | pass(대체) | donor ≠ target episode (Gate 1 §gate4 대체 충족), 판정은 사전등록 primary 1개 |
| 5 | Rollout pooling | pass | per-record×K 정렬 이식, pooling 없음 |
| 6 | Phase/dwell | pass | t0·donor_start 모두 first_grasp phase-정렬 (사전 동결) |
| 7 | Obs ≠ causation | pass | 실험 자체가 개입 + paired 검정 |
| 8 | Scene-local | pass(한정) | 결론은 "ppcc_bread 2 cell 조건부"로 한정, 일반화 주장 없음 |

**claim 등급**: intervention effect — specified donor-trajectory transplant,
cell-conditional **null**. (Gate 1 규약: 이 null은 steering 방향 전체의 폐기 근거가
아니라 "L15 full-token 통째 이식" class의 null이다.)

## 남은 질문 (미실행 — 사용자 결정 대기)

1. **짧은 창 이식** (3 record 후 자유 진행): open-loop화를 피하는 "복귀 유도" 가설 —
   primary와 다른 메커니즘이라 null의 사각.
2. **early/mid layer 이식** (L15 제외): 뒷 층들이 실제 관측을 다시 섞는 진짜 표상 개입.
3. action-replay 16%가 연 창을 쓰는 다른 개입(예: donor action 앵커 + closed-loop 혼합).
4. 성공판 이식 해악 측정 (phase-트리거 상시 개입 경로의 전제).

## 재료 위치

- 결과: `outputs/eval/robocasa/groot_n15/patchceil/<cell>/rollouts/<arm>/`, 판정
  `judge_patchceil.py`, arm 플랜 `arm_plan.tsv`, donor NPZ `donors/`(1.4GB),
  pass B 원본 pkl `passB/`(~16GB — 정리 검토 대상).
- serve 배선: worktree `exp/patching-ceiling` (`patching_hooks.py`, `patchceil_serve.py`,
  lerobot.py `--patch-*`/`/patch_arm` — 유닛 14/14·A2 실증). 미커밋.
