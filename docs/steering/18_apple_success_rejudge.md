# 18. Apple 성공 판정 재채점 (replay 기반) — 기준·도구·결과

> 2026-07-10 작성(재채점 진행 중 — §6 완료 후 갱신). 발단: steered 영상에서 "apple이 pan 위인데
> success=0" 관찰(2026-07-09). 단일출처: 이 문서 + `scripts/safe/groot_n15/robocasa/eval/rejudge_success.py`.

## 1. 판정 기준 (원인)

`PickPlaceCounterToStove._check_success()` (`robocasa/environments/kitchen/atomic/kitchen_pick_place.py:888`)
= **세 조건의 같은-step 동시 성립** (매 step 평가, chunk any/OR 집계 — 유지시간 요구 없음):

1. `check_contact(apple, pan)` — 접촉 (`object_utils.py:598`)
2. `‖apple_xy − pan_xy‖ < 0.07 m` — pan **중심** XY 거리 (`object_utils.py:599`, task가 0.07로 override)
3. `gripper_obj_far > 0.25 m` — 그리퍼 후퇴 (`object_utils.py:651`)

**pan 반경(horizontal_radius)은 ~0.23m** → 조건 2가 pan 안 안착의 상당수를 실패로 판정.
실증: fit ep43 = apple이 pan 안(중심거리 0.0706)인데 succ0 (프레임 확인).

## 2. 재채점 도구

`rejudge_success.py` — pkl의 `actions`(record당 chunk 앞 5-step 실행분)를 open-loop replay하며
매 env-step에서 [contact, 중심거리, 그리퍼거리]를 측정, **거리 임계값 grid(0.07/0.09/0.10/0.12/0.15)**로
재판정 (contact·gripper 조건은 원판정 그대로 유지). GPU·모델 불필요(CPU 물리 재생).

검증 게이트 (본실행 전 통과 필수 — [[feedback-verify-before-relay]]):
- **fidelity**: 기록 succ1의 replay 재현율 — 07-03 라운드 382/382 = 100%, 역방향 불일치 0.
- 프레임 대조: ep43(in-pan)→flip ✓, ep22/34(pan 옆)→유지 ✓.
- **머신 간 replay 재현성 OK**(실증): A100 수집분을 로컬 replay해도 fidelity 100% (같은 컨테이너
  이미지의 CPU 물리 — GPU 추론 발산과 별개).

⚠️ 함정 (실증됨):
- **한 프로세스에서 gym.make 연속 생성 시 2번째 env부터 scene 오염** → 에피소드당 fresh 프로세스 필수
  (스크립트가 subprocess 구조인 이유).
- 컨테이너를 타 호스트에서 쓸 때: 이미지 내 pip 패키지가 빌드유저 홈(.local)에 있어
  `--user <호스트uid> -e HOME=/home/<빌드유저>` + `chmod a+rX` 필요, robocasa가 asset 디렉토리에
  임시 xml을 쓰므로 asset 쓰기권한 필요.

## 3. 완료 결과 ① — 07-03/04 라운드 (ppcs_apple 단일 scene, arm 10개, 630판)

- **오판정(=pan 안 안착인데 실패 판정) 총 63/630 = 10.0%**. flip은 중심거리 0.07~0.094 구간에
  집중, **0.09~0.10에서 포화** (0.15까지 추가 flip 거의 없음).
- baseline .656→.711(+.055) / perm 계열 +.12~.15 / gated 계열 +.05~.15 →
  **원판정은 steering 해악을 계통적으로 과대평가**.
- ΔSR(vs base) 재해석: gated 계열은 교정 후에도 유의 해악(−.19~−.29) 유지, perm 계열은
  절반 축소(−.13~−.11), **perm6p60은 −.072→+.022로 부호 반전(동률)**.
- 수치: `outputs/eval/robocasa/groot_n15/rejudge_june_apple/rejudge_s*.tsv` (디렉토리명의 june은
  오기 — 실제 07-03/04 데이터).

## 4. 완료 결과 ② — fit 라벨 오염과 conceptor 영향

fit(ep0–59) 라벨 교정: ppcs_apple 43/17→45/15 (flip ep43,59) / **s100050 55/5→58/2 (flip ep28,45,55
— "실패" 5판 중 3판이 실제 성공)**. fit15는 무영향, fit30은 s100050 1판, **fit60이 오염 최대**.

연산자 비교 (`compare_conceptor_labelfix.py`, α=0.3, sanity: 재fit≡배포NPZ cosF=1.000):
- **s100050**: C_steer가 실질적으로 다른 연산자 — transport cos 0.57~0.68, λmax 0.68→0.99,
  활성차원 2→33; **place/insert-settle은 교정 후 fit 불가능**(진짜 실패 표본 부족). 판정:
  **이 cell은 대조 fit 부적격**(진짜 실패 2판) — per-seed(ps30/ps60) arm 결과에 라벨 오염 각주 필수.
- **ppcs_apple**: cos 0.86~0.97, 스펙트럼 구조 유지 — 경미.
- 수치: `outputs/eval/robocasa/groot_n15/rejudge_matrix_apple/labelfix_*.json`.

## 5. 권고 (판정 기준 교정안)

- **corrected 기준 = contact ∧ 중심거리 < 0.10 ∧ gripper_far** (flip 포화 지점, pan 반경 대비 보수적).
- 매트릭스 집계에 corrected SR **병기**(원판정 유지 — 과거 라운드와 비교 가능성 보존).
- **bread(cabinet, `obj_inside_of` th=0.05)는 별도 술어라 미검증** — apple 완료 후 같은 방법으로
  오판정 프로파일 확인 필요.

## 6. 진행 중 (완료 시 이 절 갱신)

- **매트릭스 apple 4 cell 전체**(승준 아카이브가 최완전본, cell당 16~21 arm) + fit(s100084/104) +
  구 라운드(strict/aligned/coast4 140판): 승준 20코어 5-샤드 + 로컬 1-샤드로 재채점 중
  (2026-07-10 06:41 기준 3,649/4,795 = 76%).
- 완료 후 산출: cell×arm 원판정 vs corrected SR 통합 테이블, base/steered 관계 재해석
  (confound-audit 준수), s100084/104는 A100 수집 각주.
- 실행 위치: 승준 `workspace/temporal_vla` (robocasa 이미지 28.2GB·asset 11GB 이관됨,
  컨테이너 `robocasa`), tsv는 `outputs/eval/robocasa/groot_n15/rejudge_matrix_apple/rejudge_m_s*.tsv`.
