# KAI2026 투고 원고 (제7회 한국 인공지능 학술대회)

제출 마감 **2026-09-04** · 최종본 최대 2쪽 · PDF 제출(ManuscriptLink) · Overleaf 조판.

## Overleaf 사용법

1. 이 디렉토리에서 `main.tex` + `figs/fig1_purity_residual.pdf` + `figs/fig2_granularity_margin.pdf`
   (+선택 `fig3_granularity_utility.pdf`)를 Overleaf 프로젝트에 업로드.
2. 컴파일러를 **XeLaTeX**으로 설정 (Menu → Compiler). kotex은 Overleaf 기본 포함.
3. `main.tex` 상단 TODO 3곳(저자·소속·이메일) 채우기. ACKNOWLEDGMENT 필요 시 주석 해제.
4. 현재 로컬 검증 기준 정확히 2쪽. fig3을 넣으면 초과하므로 넣을 경우 본문 압축 필요.

`preview_local.pdf` = 로컬 tectonic 검증 렌더 (한글 줄바꿈 로케일을 끄고 컴파일한
근사본 — Overleaf 산출과 줄바꿈이 약간 다를 수 있음, 참고용).

## 파일 안내

| 파일 | 내용 |
|---|---|
| `main.tex` | 원고 본체 (양식-논문샘플워드.doc 구조 대응: 국문/영문 머리블록→요약→Ⅰ·Ⅱ·Ⅲ·Ⅳ 절→References) |
| `manuscript_draft.md` | 원고의 마크다운 원본 (내용 수정은 여기와 tex 둘 다) |
| `numbers.md` | ★ 모든 수치의 정본 대조표 — 수치 수정은 반드시 여기 경유 |
| `references.md` | 서지 정본 (실존 확인 경로 포함) |
| `baseline_survey.md` / `eval_practice_survey.md` | 비교 베이스라인·eval 관행 조사 |
| `make_fig{1,2,3}.py` | figure 생성 (`~/miniconda3/envs/lerobot_safe/bin/python manuscript/make_figN.py`) |
| `figs/` | figure PDF(논문용)+PNG(눈검용) |
| `ref/` | 보강 실험 결과 JSON (게이트·잔차화·contingency) — figure 입력 |
| `양식-논문샘플워드.doc` / `양식-논문샘플한글.hwp` | 학회 공식 양식 |

## 남은 결정 (사용자)

- 저자 구성: 데이터셋 A(pq3)·전환 구조 수치는 동료(상우) 파이프라인 검증 결과 — 공저 협의
- fig3 수록 여부 (해상도–유용성 곡선; 지면 초과 시 제외)
- 사사(ACKNOWLEDGMENT) 문구
