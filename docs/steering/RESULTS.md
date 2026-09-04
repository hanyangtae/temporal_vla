# 개입 실험 결과 원장 (exp2 ~ exp5)

**이 문서 + [`results.tsv`](results.tsv) 가 SR 개입 실험 결과의 단일 출처다.**
산문·판정은 여기, arm 단위 수치는 TSV. 라운드별 원본 문서는 이 원장으로 대체됐다(§6 이력).

- 다루는 것: **활성화 개입이 SR을 바꾸는가** — conceptor / setpoint mean-diff / activation 이식
- 다루지 않는 것: SAE scene 분리(`31_`·`32_`), 표현 분석(`01_`·`08_`·`22_wrong_grasp`)
- 공통 규약: EVAL_SEED=100000 · fit↔eval episode 분리 · 위약(라벨 순열) 대조 · paired McNemar

## 0. 한 줄

**raw 대조 conceptor는 4개 라운드에 걸쳐 위약을 넘지 못했다.** 유일한 위약-분리 신호는
exp5-2의 섭동-유도 실패 회복(ppcc P1, setM DiT L10 β0.3, 위약 .25 대비 .50, 6:0 p≈.03)
하나이며 n=24 탐색 지위다. 미청산 관찰은 drawer within-instruction 약신호(2 라운드 연속).

## 1. 라운드별 판정

| 라운드 | 조건 | 규모 | 결과 |
|---|---|---|---|
| **exp2** (07-14) | scene 고정 · noise만 random · held-out ep60-119 | 115 arm / 8 cell / 60판 | **null**. pooled +6판(δ 미달). 최대 양성 s300033 +8 = 위약 +8 **동률** |
| **exp3 fit15** (07-16) | COAST 축 전부 정렬 · scene-diverse · 사전등록 | 900판 / 5 cell × 6 arm | **null**. 6-Holm 전부 기각 실패. H1 CI 상한 < +0.16 → COAST 비재현 |
| **exp3 fit30** (07-21) | fit·eval 각 2배 (검출력 2배) | 1,800판 | **null 재확정**. 게다가 **위약 요동이 ±5~6판** — null 관문 양측 위반 |
| **patchceil** (07-16) | 오라클: donor activation 통째 이식(상한 측정) | 77 target × 5 arm | **null** (2.6%, p=0.50). 단 action-replay 대조는 15.6% |
| **exp5-3** (07-27~28) | within-scene setpoint mean-diff · drawer | 160판 + A0 40셀 paired | **해악**. β=1.0 permanent −51판(구제 1 / 해악 52) |
| **exp5-2** (07-28) | 섭동으로 실패를 **유도**한 뒤 회복 | 24판/arm locked | **★유일 위약-분리 양성** (ppcc P1). 탐색 지위 |
| **per-step v4 정렬** (08-28~31) | per-step 게이팅(47) · cluster k8 phase · detector를 eval 분포(v4)로 재학습 · base-replay 재정박 paired · α=0.1 | 51케이스/118판 (pair 68) | reseed 구제 5/29·파손 5/37(유일 순효과 균형), setM 6/29·9/34, condg 1/26·4/20. ⚠ 구제 셀은 세 arm 5/5 공통 = **경계셀 속성**. ⚠ noise-0 판 결손 가능(awk 함정) |
| **v4r β sweep** (09-01~02) | 대상 scene 25판 **재수집**(캡처 ON replay, 라벨 반전 정리) → 실패 60판 × 12 arm (setm_gt/ck8 β0.6~1.0 · reseed · rsn_llr N8) | 60판 (jug 22·oven 17) | 구제율 최고 12%: rsn_llr 5/43(candle 3/8) · setm_ck8 β1.0 5/43 · reseed 3/60. jug·oven 전 arm 0~2. β 비단조. 파손 축 없음. **fit pool 타 scene은 구 세계(혼합) — 미결** |

## 2. 반복 확인된 사실 (라운드를 넘어 재현된 것)

1. **위약이 처치와 같거나 그 이상 움직인다.** exp2 s300033 +8 = 위약 +8. exp3 fit30에서
   위약 자체가 drawer +6 / ppcc −5. exp5-3 future-only도 위약 미초과. **개입 크기 자체가
   결과를 재추첨**하고 있다는 뜻이다.
2. **net 뒤에 양방향 대량 flip.** exp2 s300033은 +8을 만들려고 20판 살리고 **12판을 죽였다**
   (60판 중 53% 뒤집힘). 전 cell에서 성공→실패가 2~12판씩 발생. "실패 구제"가 아니라 re-roll.
3. **해악은 용량 문제.** exp5-3 β 단조 회복(1.0 → 0.5 → 0.2에서 SR .025 → .225 → .300).
4. **파괴의 주범은 action 토큰 개입.** 같은 β=1.0에서 full 0.025 vs future-only 0.350
   (jerk 0.94×). 세그먼트를 가리면 무해해지지만 **동시에 이득도 사라진다**.
5. **read ≠ write.** 진단에서 AUROC 0.84가 읽히는 방향으로 밀어도 SR이 안 오르거나 무너진다.
6. **천장·바닥 cell은 검출력이 없다.** SR 0.97(apple_s100422, flip 2판) / 0.93(pizza_cutter).

## 3. 미청산 관찰 (기각도 채택도 안 된 것)

| 관찰 | 근거 | 왜 채택 못 하나 |
|---|---|---|
| **drawer within-instruction 이득** | fit15 gated 7W/1L(+6), fit30 perm 16W/5L(+11)·gated 18W/9L(+9) — 2 라운드 연속 방향 유지 | 같은 라운드 위약이 +6. 순증 +3~+5, 미보정·탐색. **PPCC에서 비재현**(−5~−6)이라 task-일반 근거 불가 |
| **exp5-2 ppcc P1 회복** | setM .50 vs 위약 .25, paired 6:0 p≈.03 | n=24, 다중비교 보정 없음, 단일 cell·단일 섭동. 위약이 완전 class-blind 아님(VL heldout AUROC 0.82) |
| **개입으로 뒤집을 창은 존재** | patchceil action-replay 15.6%(s400020 22%) | 오라클 정보(donor 궤적) 사용 — 배포 가능한 방법이 아님. "실패의 다수는 grasp 진입 민감성"의 근거 |

## 4. COAST 대조 — 무엇이 달랐나

| 항목 | COAST | 우리 |
|---|---|---|
| 평가 무대 | scene 매 episode 랜덤 재샘플 | exp2 = scene 1개 고정 / exp3 = scene-diverse |
| pairing | 조건별 fresh rollout (비-paired) | ep-paired |
| denoising | step별 벡터 개별 stack | K개 mean-pool (exp3에서 정렬) |
| 토큰 풀링 | 49토큰 전체 | action 16토큰 (exp3에서 정렬) |
| GR00T×RoboCasa | mean ΔGlob **+0.16** | exp3 H1 CI 상한 +0.117 / +0.022 |

- exp3에서 **축을 전부 정렬**하고도 비재현 → "풀링·정렬 차이 때문"이라는 설명은 소진됐다.
- COAST는 **성공→실패 flip을 보고하지 않는다** — 랜덤-scene·비-paired 설계로는 계산 자체가 불가.
  우리 §2-2의 churn은 그 방식으로는 보이지 않는 정보다.
- 우리가 고른 두 task는 COAST 자체에서도 최약체(PP Cabinet +0.07 = 7 task 중 공동 최하위,
  PP Stove +0.14). 단 헤드룸 있는 cell에서도 예측치가 안 나와 **헤드룸만의 문제는 아니다.**
- 확정 가능한 범위: **"우리 조건에서 미재현"**. "COAST가 틀렸다"까지는 아니다.

## 5. 데이터 위치

arm별 경로는 `results.tsv`의 `데이터` 열. 루트는 `outputs/eval/robocasa/groot_n15/`.

| 라운드 | 판정 산출물 | 원료 activation |
|---|---|---|
| exp2 | `steer_eval_exp2/aggregate_v2/` (arms.tsv·matrix.md·summary.json) | ⚠ **5 cell fit 원료 pkl 소실** — 3단 연쇄 사고 |
| exp3 fit15 | `steer_eval_exp3/aggregate_f/` (decision_sha `ab0e3abd59ddf52c`) | eval 캡처 OFF |
| exp3 fit30 | `steer_eval_exp3/aggregate_f30/` (동일 sha) | eval 캡처 OFF |
| patchceil | `patchceil/<cell>/rollouts/<arm>/` + `donors/`(1.4GB) | `passB/`(~16GB, 정리 검토) |
| exp5-3 | 승준 `~/sm_*{drawer,mixer}*.json` · `~/exp53_npz/fit_report.json` | 승준 아카이브 |
| exp5-2 | `exp42_induced/` (전부 로컬) | 로컬 |
| per-step v4 정렬 | `og_ck8v4_expand{,_srv50}/<arm>/<case>/` · 매니페스트 `outputs/steer/online_pipe/manifests/v4_expand_eval.tsv` | eval 캡처 OFF |
| v4r β sweep | `og_v4r_expand/<arm>/<case>/`(kanu·srv50·srv48 분산, 회수본 `outputs/tmp/v4r_results/`) · 정본 `manifests/v4r_labels.tsv`(160)·`v4r_eval.tsv`(60) | 재수집 `og_v4r_collect/`(pkl, 승준 `v4r_collect/` 복제) · eval은 7-layer hook ON·미저장 |

> 두 라운드의 좌표·함정·판정 서사는 [`../collab_within_claude/handoff_20260902_전체파이프라인.md`](../collab_within_claude/handoff_20260902_전체파이프라인.md).

> ⚠ 2026-07-14 eval activation pkl/zst 전 호스트 삭제(~172GB). 판정 sidecar·fit·conceptor는 보존.
> exp3부터 eval 캡처를 끈다. 데이터 소실의 재발 방지 규약은 [`docs/04_data_storage_convention.md`](../04_data_storage_convention.md) §7.5–7.6.

## 6. 흡수 이력

이 원장이 대체한 문서(내용은 여기로 이관, 원본은 git 이력에):

| 원본 | 내용 |
|---|---|
| `23_exp2_scene_fixed_steering_results.md` | exp2 8 cell · flip 분석 · apple 채점 불변성 |
| `21_exp3_results.md` | exp3 fit15 사전등록 6가설 + 탐색 arm |
| `22_exp3_fit30_results.md` | exp3 fit30 1800판 |
| `25b_patchceil_transplant_result.md` | donor 이식 null + action-replay 대조 |
| `36_exp5-3_within_scene_steer.md` | drawer setM · β sweep (mixer 수집분은 별도) |
| `28_exp4-2_p0_report.md` | 섭동 메뉴 확정 · 유도 실패율 게이트 (SR 개입 아님 — 전제 기록) |

`35_exp5-2_results.md`(섭동-유도 실패 회복, 2026-09-02 archive)의 요지는 §1·§3 행에 있다 —
무대가 다름(clean 실패가 아니라 유도된 실패)을 잊지 말 것. 상세는 git 이력. `31_`·`32_`는 SAE 표현 분석이라 이 원장에 넣지 않는다.
