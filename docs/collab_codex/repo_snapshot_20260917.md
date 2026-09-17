# 2026-09-17 원고·자료 정리 기록

## 보존 범위

- `manuscript/main.tex` 그림 배치 수정, `make_docx.py` 수식 문자 변경과 기존 초안 4개 삭제를 반영했다. 원고 내용·연구 수치는 이번 정리에서 수정하거나 재검증하지 않았다.
- `manuscript/main.pdf`, `main.docx`, `인공지능학술대회.pdf`, `figs/*_300.png`는 기존 파일 그대로 저장한 산출물이다. 현재 소스에서 다시 생성한 결과라는 보장은 없다.
- 루트 `IEEE-RAL-Appendix.pdf`, `supplementary.tex`, `resolution_timeline.png`, `로봇 AI 대기업 목록.pdf`, `vla_intern_interview_questions.md` 및 `docs/references/PPS.pdf`는 기존 참고 자료로 보존했다.
- `supplementary.tex`가 참조하는 루트 `figs/` 이미지 5개는 이 작업 폴더에 없다. 따라서 독립 빌드 가능한 논문 패키지로 간주하지 않는다.
- `v6_failure_onset_labels.tsv`는 로컬에 있던 266행 라벨 스냅샷이다. 라이브 라벨 정본이 아니며, 원본 rollout과 다른 머신의 재수집 영상 라벨을 임의로 동일시해서는 안 된다. 최신 라벨 여부와 연구 결과 정합성을 이번에 검증하지 않았다.
- CosmosPolicy grid feasibility 보고서는 기존 분석 문서 그대로 보존했다. 현재 환경에서 재실행한 결과가 아니다.

## 좌우 비교 실험 보관

메인 코드·설정·테스트에서 `lr_equal_budget` 경로를 참조하는 외부 실행 코드는 검색되지 않았다. 관련 runbook과 해석 보고서만 참조했다. 코드·설정 16개는 `exp/lr-equal-budget-20260907`의 `5f2c64a`와 바이트 단위로 동일했다.

메인 작업 폴더의 미추적 사본은 `outputs/repo_cleanup_20260917/` 아래 원래 상대 경로로 이동했다. 해당 브랜치는 유지했다. 브랜치와 다른 runbook·해석 보고서, 임시 `live.main.tex` 및 `manuscript/chk/`도 같은 위치에 보관했다. `manifest.json`에 보관 파일 SHA256을 기록하고 이동 후 일치 여부를 검증했다. 이 로컬 보관 경로는 Git 추적 대상이 아니다.

## 검증 범위

- 원고 그림 파일 존재, label/ref 대응, Python 문법 검사.
- DOCX ZIP CRC와 본문 XML 파싱, PNG 무결성, PDF 시작·종료 표식 검사.
- 라벨 TSV 열 구조 및 marks JSON 파싱.
- `git diff --check`.

XeLaTeX/PDF 재조판 및 DOCX 재생성은 수행하지 않았다. 위 검사는 PDF의 모든 페이지가 정상 렌더되는지 또는 수치가 맞는지를 증명하지 않는다.

정리 과정에서 supplementary.tex의 줄 끝 공백과 라벨 TSV의 CRLF 줄바꿈만 정규화했다. TSV 필드 값은 변환 전후 동일함을 확인했다. PDF는 바이너리로 명시해 원본 바이트를 유지한다.
