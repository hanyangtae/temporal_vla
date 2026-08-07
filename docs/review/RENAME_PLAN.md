# 전역 재배치·명명 대장 (검토 중 누적, 전 스테이지 완료 후 일괄 적용)

> 설계 원칙(2026-07-28 설계 문서 §6): 국소는 즉시, **전역 rename 은 마지막에 일괄**.
> 여기는 그 일괄 적용의 입력이다. 스테이지 진행 중 keep 판정이 났지만
> "라운드 디렉토리에 사는 라운드-무관 도구"를 발견할 때마다 추가한다.

## 명명 원칙 (사용자 확정 2026-08-07)

**exp 번호가 아니라 기능명으로 관리한다.** exp 숫자는 라운드마다 바뀌지만 기능은
반복된다 — 대표 사례: exp4-2 의 "induced" → **perturb** (교란 실험은 상설 축).

## 목표 배치 — 두 개의 새 홈

- **`scripts/collect/`** — 수집·무결성 (씨앗: `verify_grid.py` 已존재)
- **`scripts/fit/`** — 연산자 생성·진단

## 재배치 후보 (keep 판정분)

| 현 위치 | 이동 후보 | 근거 |
|---|---|---|
| `steer/exp4_1/fit_mean_diff.py` | `scripts/fit/fit_setm.py` | 메인 연산자(setM) fit |
| `steer/exp5_3/fit_within_scene_setM.py` | `scripts/fit/fit_setm_within_scene.py` | 현행 표준 fit |
| `steer/exp5_3/fit_placebo.py` | `scripts/fit/fit_placebo.py` | 위약 대조 상설 |
| `steer/exp5_3/make_fut_npz.py` | `scripts/fit/` | token 선택 부속 |
| `steer/fit_phase_conceptor_n15.py` | `scripts/fit/fit_phase_conceptor.py` | n15 접미어도 제거(프로파일화와 함께) |
| `steer/diag_conceptor_nondegen.py` | `scripts/fit/diag_nondegen.py` | 연산자 비퇴화 진단 |
| `steer/induced/induced_common.py` | `scripts/fit/load_lib.py` (가칭) | 범용 pkl 로더 디스패치 — diag(keep)가 의존 |
| `steer/induced/` 나머지 10개 | `scripts/perturb/` (가칭) | perturb 상설 축 — "induced" 명칭도 perturb 로 |
| `steer/exp5_4/probe_lib.py`·`smoke_probe.py`·`check_probe_identity.py` | `scripts/collect/` | serve 무결성 도구 |
| `analyze/mixer_scene_feasibility.py`·`drawer_scene_feasibility.py` | `scripts/collect/` | 수집 전 scene 필터 (S3 keep) — 복사-수정 쌍이라 이동 시 공용화 검토 |
| `n15/collect/build_perturb_grid.py`·`collect_perturb_grid.sh` | `scripts/perturb/` | S1 keep — perturb 축 합류 |

## 이동 시점 규칙

- 수집 세션이 참조 중인 파일(feasibility 등)은 **라운드 완주 후**.
- import 수정은 일괄 1회 (S5~S7 keep 추가분 합류 후).
- 이동 시 각 파일의 라운드 유래는 파일 상단 주석에 1줄 남긴다 (출처 보존 관행).
