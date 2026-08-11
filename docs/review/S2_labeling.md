# S2 — phase 라벨링 스테이지 카드 (2026-08-07)

기계 판독분: [`S2_files.tsv`](S2_files.tsv) · 판정 UI: `python3 scripts/review/ledger_ui.py`

## 규모

12파일 중 **판정 완료 3** (S1에서 keep: event_labeler·step_phase·phase_live_render),
실질 판정 대상 **9파일 ≈ 1,450줄**. S1 대비 작다 — 핵심(1,239줄 라벨러)은 이미 읽었다.

## 구조 한 장

```
libero/event_phase_labeler.py  ← 순수 분할 코어 (PhaseSegmenter, 상태 기반·비단조)
    ↑ 재사용                        + LIBERO detector (find_domain ← bddl_phase_labeler)
robocasa/event_labeler.py      ← robocasa 술어 detector + task군 라벨러 3벌 (S1 keep)
    ↑ 별도 인스턴스
robocasa/step_phase.py         ← env-step GT (S1 keep)
eval/env_step_gt_retro·batch   ← 같은 라벨러를 소급 replay 로 (구 데이터용)
src/phase_online/online_phase.py ← 추론 중 readout (serve/lerobot.py 소비, S5 걸침)
```

## 판정 축 — 사실상 질문 2개

1. **libero 축을 유지하나?** `bddl_phase_labeler`(145) + `collect_pi05_libero.sh`(111) +
   HANDOFF(154) 는 LIBERO 전용이다. 단 `event_phase_labeler`(332) 는 코어라 robocasa 가
   물고 있음 — libero 를 접어도 코어는 남고, LIBERO detector 부분(~120줄)만 분리 대상.
   (참고: phase-selective steering 계획이 GR00T+libero10 을 첫 실험으로 잡았던 이력 있음)
2. **소급 GT 도구(retro·batch, 330줄)를 유지하나?** 구 데이터 전량 폐기로 "기존 pkl 에
   소급"이라는 원래 용도는 사라짐. 신규 수집은 실시간 GT 기록. 남는 용도는
   "라벨러를 고친 뒤 이미 수집된 판에 재라벨" — 재수집 중 라벨러가 바뀔 가능성을
   어떻게 보느냐에 따라 keep/archive 가 갈린다.

## 알려진 사실 (S1 정독에서 이월)

- `TASK_EVENTS` 미등록 task 는 수집이 KeyError 로 죽는다 — 재수집 8종 중 3종 미지원
  (docs/steering/38 §2.0 블로커, 수집 세션 처리 중).
- factory 분기가 문자열 `in` 매칭이라 순서 의존 (`FridgeDrawer` 명시 제외 등).
  task 가 늘수록 함정 — 신규 task 등록 절차를 문서화할 가치.
- 라벨은 두 벌 저장된다: record 단위(activation 1:1, gated steering 용) +
  env-step GT (사후 분석·t0 주석용). QA 열 `env_step_record_mismatch`.
- 테스트 커버: 라벨러 계열 5벌 54개 + libero 코어/bddl — groot 컨테이너에서 실행.
