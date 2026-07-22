# patchceil 사전등록 규약 (v2.1 — 2026-07-16 동결)

본 실행 결과를 보기 전에 동결하는 판정 규칙. 변경 시 사유·일자를 아래 "개정 이력"에 기록
(무단 변경 = 판정 무효). 설계 근거: `docs/collab/2026-07-16-patching-transplant-gate1.md`,
plan `~/.claude/plans/pq3-wise-mist.md`.

## 대상·donor

- targets = `patchceil/<cell>/targets_fit.tsv` 의 실패 **전량** (s300033 40 + s400020 37 = 77).
- donor/placebo/sham = `passB_manifest.tsv` (select_passB.py 결정적 규칙: donor 4 =
  grasp·insert-settle 도달 성공 중 길이 중앙값 근접, placebo 4 = 실패 idx 균등간격,
  sham = placebo 앞 2판).
- **donor 배정**: target ep 오름차순 → donor 4개 round-robin. placebo·shuffle arm 도
  같은 순환으로 대응 identity 유지 (Gate 1 §채택6).

## t0 (분석 재료: patchceil_meta ep*.json, 전부 baseline 메타 — patched run 트리거 금지)

- **primary t0 = first_grasp**: target baseline `feature_phases` 에서 'grasp' 최초 등장
  record index. 근거: 전 target 도달(77/77, record 4~8 — 선택편향 0, ITT 안전).
- donor 정렬: `donor_start` = donor 의 first_grasp record (phase-시작점 정렬 v1).
- exploratory t0 (primary 유의 시에만): reach 초입=record 2 / first_place
  (미도달 target 은 ITT 비구제 + eligible-조건부 병기) / late=record 30.
- env-step→record 변환이 필요한 경우 **floor** (record r 은 env-step [5r, 5r+5)).

## 개입·창

- **primary arm**: layer **L15** (DiT 마지막 block residual, D=1536), token_select=all
  (49-token), K=4 정렬 재생, `patch_len=-1` (donor 고갈까지 — 고갈 후 합성/freeze 금지,
  자유 진행).
- 채점: ① episode 성공(주 지표, 파일명 succ — bread cell 이라 rejudge 불요)
  ② 패치 창 내 성공 (`first_success_step` ≤ 5×(t0+창길이), 병기).
- exploratory: W short=3 records / layer early/mid = {0,2,4,8,10,12} (L15 제외) /
  cross-cell shuffle.

## 대조 arm

1. **no-patch**: 같은 (scenario_seed, inference_seed=ep×1000) 재실행 — paired base.
   결정론상 실패 재현이 기대값 (불일치 ep 는 판정 제외·보고).
2. **placebo-fail**: 대응 placebo donor(실패)의 activation 동일 방식 이식.
3. **donor-shuffle**: 대응 success donor 의 record 순서를 고정 seed(20260716) permute
   (K 축 정렬은 유지) — "성공 정보"와 "임의 강한 교란" 구분.
4. **direct action-replay**: activation 대신 donor 의 저장 action 을 t0 부터 open-loop
   주입 (sim only). 양성이 activation-국소화인지 action 재생인지 구분.

## Anchor (본 실행 전 통과 필수)

- A1 결정론: pass B 재수집 16판의 succ **및 actions 수치**가 승준 zst 원본과 일치.
- A2 action-equivalence: target 1판을 inference_seed=donor 것으로 재실행 + L15 전창
  patch → emitted actions == donor 저장 actions (fp16 tol). 배선 유닛테스트.
- A3 sham: sham 2판 — 자기 activation 이식 rollout == baseline 완전 일치.

## 판정 (primary 1개, hierarchical)

- 구제 = target(원 실패)이 해당 arm 에서 성공.
- **primary statistic**: donor-success arm vs {placebo-fail, donor-shuffle} 각각 paired
  (target ep 단위) exact McNemar one-sided (`pq3_decision.exact_mcnemar_one_sided`
  import) → **p_final = max(p_vs_placebo, p_vs_shuffle)**, α=0.05, cell-stratified 합산
  + cell 별 병기.
- primary 비유의 → exploratory 미실행 (규모 자체가 결론). 유의 → t0 곡선·layer 분해.
- claim 등급 고정: "intervention effect — specified donor-trajectory transplant,
  cell-conditional". 구제율 0 이어도 steering 방향 폐기 결론 금지.
- 보고 전 confound-audit 스킬 표 필수 (N/A 사유 명시).

## 개정 이력

- 2026-07-16 v1 동결 (Gate 1 + 데이터 정정 반영, GPU=로컬 1번 사용자 승인).
