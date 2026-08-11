# S7 — 집계·판정·통계 스테이지 카드 (2026-08-10)

기계 판독분: [`S7_files.tsv`](S7_files.tsv) · 판정 UI: `python3 scripts/review/ledger_ui.py`

## 규모 — 44파일 7.8k줄. S6 과 같은 무리 구도, 단 성격이 다르다

이 파일들은 러너가 아니라 **RESULTS.md 숫자의 재현 경로**다. 사이드카/TSV 를 읽어
통계(Holm·McNemar·순열·위약 대조)를 내는 쪽 — 지우면 "그 숫자를 어떻게 얻었나"의
실행 가능한 증거가 git 이력으로만 남는다.

| 무리 | 파일 | 상태 |
|---|---|---|
| exp2 집계·게이트 | 4 | 종결(null) |
| exp3 집계·게이트 (6-Holm·CI 규칙 구현) | 6 | 종결(null) |
| exp4-1 집계·감사 | 4 | 완료 |
| exp5-3 집계 | 1 | 현행연계 — 차기 판정의 직계 조상 |
| exp5-4 선별 통계 (순열·위약·검정력) | 9 | 종결(seed암기) |
| perturb 집계·smoke 판정 | 2 | keep 축 |
| patchceil 판정 일습 | 9 | 보조종결 |
| 재채점·eval 부품 (`eval/`) | 7 | ★현행 2 포함 |
| 루트 잡 | 2 | 종결 |

## 판정 축 — 질문 2개

1. **종결 라운드 집계를 S6 러너처럼 일괄 archive 하나?**
   S6 과 다른 점 — 러너는 "다시 돌릴 일 없음"이 명백했지만, 집계는 **아카이브에
   남은 사이드카를 다시 읽어 숫자를 재검산할 유일한 도구**다 (예: 논문·보고서 작성
   때 "exp3 null 의 Holm 값 다시 뽑아줘"). git 복원이 가능하긴 하다.
   절충안: 통계 패턴(위약·Holm·paired·순열)의 **공용 증류본을 만들고** 라운드
   전용본은 archive — S6 의 exp5-3 용례 문서와 같은 방식, 단 이번엔 문서가 아니라
   재사용 가능한 판정 라이브러리(차기 스테이지 판정에 바로 쓸 것이므로).
2. **`eval/` 7개의 개별 처분** — rejudge·lerobot_http_eval 은 ★현행 keep 자명.
   native ZMQ eval 2개는 COAST 시대 유산(충실 재현용)인데 그 라운드는 종결.
   annotate_phase_video 는 수집 세션이 쓰는지 확인 필요 (mixer 러너가 참조했었음).

## 참고

- lerobot_http_eval.py 는 S1 허브의 import 대상 — 사실상 지도 1 의 부품인데 eval/
  에 산다 (RENAME_PLAN 후보).
- exp3 gate 에는 전용 테스트(test_gate_a)가 동반 — archive 시 함께.
