# S3 — split·manifest·feasibility 스테이지 카드 (2026-08-07)

기계 판독분: [`S3_files.tsv`](S3_files.tsv) · 판정 UI: `python3 scripts/review/ledger_ui.py`

## 규모 — 10파일 1,906줄. 구도가 단순하다

두 무리뿐이다:

**① SAFE-detector 시대 split 8파일 (05-28~06-10, 1,406줄)**
seen18/seen6 rollout 을 LSTM 실패검출기 학습용 train/CP/test 로 나누던 도구.
`split_lib.py`(공용) + n15 3개 + n16 3개 + lerobot 1개.

- 대상이던 구 stem rollout 은 **전량 폐기**됐다 (eval pkl purge + 재수집 결정).
- 새 좌표 그리드에서 fit/eval 분리는 **seed 단위**로 한다 (fit-seed ↔ eval-seed 분리
  표준, in-sample rescue 방지) — symlink split 방식 자체가 구세대.
- 단 lerobot `prepare_split.py` 는 특정 라운드 비종속(범용 manifest 생성)이라 성격이 다름.

**② scene 실현가능성 필터 2파일 (08-05~06, 500줄) — ★수집 세션이 지금 쓰는 중**
mixer(원조) + drawer(이식판). 정책 무관 관절 스윕으로 "기하학적으로 성공 불가능한
scene seed" 를 fit·eval 양쪽에서 동일 제외 (`SCENE_FEASIBILITY.md`).
**08-06 에도 수정됨 — S1 허브와 같은 이유로 이번 라운드 완주까지 동결.**

## 판정 축 — 질문 2개

1. **SAFE split 8개를 남길 이유가 있나?** 근거 후보: (a) SAFE-LSTM 검출기 재학습
   가능성 (검증된 결론: 공정 metric 에서 unseen ~chance — 재개 동기 약함),
   (b) 아카이브의 seen18 판정·sidecar 를 다시 나눌 일. 없다면 8개 일괄 archive 가
   자연스럽고, 살릴 가치가 있는 건 범용인 lerobot `prepare_split.py` 정도.
2. **feasibility 2개의 자리.** 판정은 keep 이 확실한데 위치가 `analyze/` 다 —
   역할은 S3(수집 전 필터)라 라운드 종료 후 `scripts/collect/` 이동 후보.
   또 mixer→drawer 복사-수정 패턴이라 3번째 task 가 생기면 공용화 시점.

## 스테이지 밖으로 보낸 것

`steer/exp*/make_*_manifest.py` 5개 — fit 재료 선별용 manifest 로 역할은 비슷하지만
라운드 전용 fit 파이프라인 소속이라 S4(fit)에서 그 라운드 유산과 함께 판정한다.
