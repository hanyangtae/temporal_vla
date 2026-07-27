# exp5-2 결과 — 섭동-유도 실패의 activation steering 회복 (ΔSR)

작성 2026-07-28. 계획·배경 = [`29_exp5-2_handoff.md`](29_exp5-2_handoff.md). 실행 머신 =
dongkyu 원격(pdk_external, RTX 4070 Ti SUPER 16GB 1장; **bitwise 비결정 — 첫 편차 ≤5e-3
규약으로 판정**). 브랜치 `exp/exp5-2-perturb-steering` (dev PR#69 이후).

★ 본 라운드는 탐색적(defer-rigorous-validation): 위약·paired는 수행했으나 다중비교 보정
없음, n=24/arm. "회복 성립" 주장은 confirmatory 재실험 전까지 잠정.

## 0. 한 줄 요약

**ppcc_bread P1(물체 displace)에서 처음으로 위약-분리 회복 신호** — locked 24판에서
setM(DiT L10, β0.3) SR .50 vs 위약 .25 vs OFF .33, paired discordant ON:위약 = **6:0**
(McNemar exact p≈0.03). C1(카메라)은 DiT setM이 해악·**VL 평균이동(setpoint_vl)** 이 정합
(ppcc 약양성 3:0). 그 외 섭동·cell 대부분은 구조적 사유로 탈락(§3).

## 1. 방법 요약

- fit: clean(baseline_cap succ) vs perturbed(캡처, record_start 절단) **평균차이** —
  r̂=normalize(μ_clean−μ_perturbed), s=μ_clean·r̂, per-episode 균등 k=20 서브샘플,
  **fit=짝수 ep / eval=홀수 ep(locked)** 분리. DiT는 setpoint_seg(action 토큰 33:49만),
  VL은 **setpoint_vl 신규 op**(vlln 토큰평균 이동 — 토큰 내 분산 보존, `scripts/serve/
  steering_hooks.py`).
- 위약 setM_pl: episode 라벨 순열 + **dose-match**(DiT=seg_mask 게인, VL=setpoint 평행이동
  — serve가 게인을 곱셈 적용함을 코드로 검증). ⚠️순열 위약 방향이 완전 class-blind 아님
  (VL heldout AUROC 0.82) — 위약은 "무효"가 아니라 "약화된 방향".
- eval: exp4-2 grid.tsv 행 재사용으로 (scene seed, inference seed, spec_seed) arm 간 자동
  paired. P1/P2는 trigger latch(`--steer-from-record`), C1/G1은 전 구간.

## 2. ppcc_bread 파일럿 (scene 100084, 원격)

fit-split 스윕(짝수 12판)과 locked 3-arm(홀수 24판):

| 섭동 | arm | SR (locked 24판) | paired |
|---|---|---|---|
| p1_d003 | OFF | 8/24 (.33) | — |
| p1_d003 | **setM DiT L10 β0.3** | **12/24 (.50)** | ON:OFF = 6:2 |
| p1_d003 | setM_pl (위약) | 6/24 (.25) | **ON:위약 = 6:0 (p≈0.03)** |
| c1_s200 | OFF | 8/24 (.33) | — |
| c1_s200 | setm_vl VL β1.0 | 9/24 (.375) | ON:위약 = 3:0 |
| c1_s200 | setm_vl_pl | 6/24 (.25) | — |

- 위약이 OFF보다 낮은 패턴 2회 반복 — 방향 없는 동일-크기 개입은 무익/해악, 즉 효과가
  크기 아닌 **방향**에 실림.
- c1 DiT setM은 용량-의존 **악화**(OFF 3/12 → β0.3 2/12 → β1.0 1/12, 짝수 12판) — C1은
  Δ가 확산형(top1 EVR .33–.60)이라 rank-1 DiT 조준이 어긋남. VL 평균이동이 정합.
- p2_f040d2: DiT L12 β{0.3,1.0}·L10 β1.0 전부 3/12 = OFF와 동률 — 무효.

## 3. 탈락 목록과 사유 (grid 캘리브레이션·게이트 실측)

| 항목 | 사유 |
|---|---|
| mixer P1 (head 관절 displace) | task-자유도 confound: +방향=head 열림=task 진척(실패 0%), −방향=닫힘 리밋 클램프 |
| drawer P1 d020 | **로봇 도달 전 서랍 열림 → 로봇 끼임** (사용자 실화면 확인, 데이터 폐기) |
| drawer P1 (전 세기) | 10→12cm 실패율 절벽(25%→92%) — 창 없음. d012 hard는 fit 게이트 탈락(아래) |
| mixer P2 (전 세기) | 40N도 head 조작 방해 못함 — 캡처 실효 실패 ~20%, grid 50%는 n=4 노이즈 |
| mixer/drawer G1 | 분리 수치는 있으나 **outcome-matched(clean 전라벨) 대조 시 chance 붕괴** = 성공/실패 길이 아티팩트 (seen18 길이 confound 재현) |
| ppcc G1 | 분리 자체 없음 (exp4-2, 전층 AUROC .70–.82) |
| C1-VL conceptor | 스펙트럼 비퇴화여도 클래스 선택도 없음(균일 norm 수축 M≈(1−β)I) — VL 분리가 평균이동이라 2차모멘트 연산자 부적합 → setM-VL로 대체 |

## 4. cell 확장 결과 (drawer_left · mixer)

fit 분리도 게이트 = held-out(홀수 ep) episode AUROC ≥ ~0.96(=ppcc p1 수준) **AND**
outcome-matched(clean 전라벨 60판) 대조에서 잔존:

| cell | cfg | 주 fit (best) | outcome-matched | 판정 |
|---|---|---|---|---|
| mixer | c1_s100 | DiT .93–.95 / **VL 1.000** | .78 잔존 | 분리 실재 |
| mixer | g1_x005 | L12 .960 | **.57 붕괴** | 아티팩트 |
| mixer | p2_f040d2 | .61–.72 (n=10) | 방향 역전 | 무정보 |
| drawer | c1_s025 | DiT .81 / **VL .914** | VL .685 잔존 | 경계선 |
| drawer | g1_x005 | .837 | chance 붕괴 | 아티팩트 |
| drawer | p1_d012 | .74 (n=12) | **.29 역전** | 무정보 |

**mixer ΔSR eval (c1_s100, VL β1.0, locked 24판)**: OFF 1/24 · setm_vl 0/24 · 위약 1/24 —
**base SR floor로 판정 불가**. mixer clean SR .167, 섭동 하 .06 → 회복 창 +0.10뿐인데
자연 실패가 83%. **mixer는 분리 근거로만 사용** (사용자 확정 07-28). grid의 c1_s100 50%
실패율(n=4)은 cal-pool 편향+노이즈였고 신선 ep 포함 실측은 94% 실패.

drawer는 게이트 전 config 탈락(DiT) / VL 경계선(.914). 개선안(§6) 사용자 결정 대기.

## 5. 함정 대장 (이번 라운드 실측)

1. **원격 결정성**: dongkyu 원격은 bitwise 재현 안 됨(record ~118부터 run-to-run 발산,
   GPU 비결정) — sham/짝 검증은 첫 편차 ≤5e-3 규약. drawer는 outcome까지 뒤집힘(grid
   실패율 .58 → 캡처 .65).
2. **clean-succ vs perturbed-timeout fit은 길이/outcome 교락** — G1 두 cell이 이걸로
   허위 통과할 뻔함. **outcome-matched 강건성 점검을 게이트에 상설**.
3. task-자유도(fixture 관절) 섭동은 방향 양쪽 다 confound 위험(진척/리밋/끼임) —
   P1은 자유물체 cell(ppcc)에서만 깨끗.
4. grid n=4는 실패율 추정에 부족(mixer p2 50%→실측 20%) — 채택 전 n=12 재측정 필수.
5. 반복 행 (config,ep) 충돌: 캘리브 풀 < 요청 판수면 C1/G1은 신선 ep 확장, P1/P2는 풀
   clamp (build_perturb_grid fix, 03e28be).
6. serve 2개 + steering(VL)은 16GB에서 CUBLAS 크래시 — VL arm eval은 serve 1개.
7. bash: `${x:+VAR=v}` prefix-대입 오파싱(rc=127) → `env` 인자로; ssh setsid 발사는
   `< /dev/null`; `GROUPS`/`pgrep -f` 자기매칭.

## 6. 다음 (미결)

- **drawer 개선 패키지** (사용자 결정 대기): clean baseline 60→180판(+2.5h) + C1 세기
  s050 캡처 48판(+1.5h) → 재fit·게이트 → 통과 시 3-arm eval(+2.5h). 회복 창 ≈ +0.18.
- ppcc p1 양성의 confirmatory 재실험(새 scene/seed, 사전등록) — 본 라운드는 탐색적.
- ppcc c1 setm_vl은 β/개입 창 스윕 여지(3:0, 약양성).

## 7. 데이터 위치

- 원격(pdk_external): `outputs/eval/robocasa/groot_n15/exp52/` (ppcc fit NPZ·eval),
  `exp52_induced/{drawer_left,mixer}/` (baseline·grid·capture·fits). 승준 HDD 일괄 전송 예정.
- kanu: `outputs/eval/robocasa/groot_n15/exp52/fits_ppcc/` (ppcc fit 원본, VL conceptor 포함).
