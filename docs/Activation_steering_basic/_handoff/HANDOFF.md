# Handoff — Activation Steering 종합 정리 작업

작성 2026-07-02. 이 문서 하나로 작업을 이어받을 수 있게 정리. 검토(리뷰) 도중 핸드오프로 전환됨.

> **[2026-07-02 후속 세션에서 처리 완료 — 이 핸드오프는 종결됨]**
> - §2 결함 A/B 수정 + 재빌드 완료. 추가로 **결함 C**(변환기가 `_..._`를 이탤릭으로 소비해
>   `C_success`·`SAE_VLA_pi05.md` 같은 snake_case 수식/파일명이 깨짐)를 발견·수정
>   (`notion_push.py`: continuation-line 병합 + `>` 인용 병합 + underscore 이탤릭 규칙 제거).
> - §3 남은 검토 완료: 전 11페이지 렌더 스윕(리터럴 `**` 0, snake_case 훼손 0), 표본 15편
>   노트↔PDF 원문 대조(수치·수식·인용 오류 0), 논문 선정 평가(플랜 전 축 커버 확인).
> - 선정 공백 보강: **#52 RepE Survey**(Wehner, TMLR 2025, arXiv 2502.19649) 추가 —
>   PDF·notes/RepESurvey.md·서베이 §1/부록B·Notion 반영. 이제 정독 52편.
>
> **[2026-07-02 스타일 개편 — Notion 단일 페이지 재구성]**
> - 사용자 요청: robots-oh 주간리포트 2주차처럼 h1/2/3·toggle·bold·highlight로 가독성↑.
> - 서베이 md를 3단계 heading 구조로 재작성: `#`=대주제(§1~§7·요약·용어·부록, red h1),
>   `##`=소주제(정의/대표논문/우리연결 등, green h2), `###`=§5 서브라벨(blue h3),
>   `**면접 포인트.**`→`<details>` toggle(🎤), 핵심 논지 3줄 `==highlight==`(yellow).
> - **Notion은 이제 11 child page가 아니라 부모 페이지에 통째로 렌더되는 단일 리치 페이지.**
>   `refresh_notion.py`가 single-page 빌드로 재작성됨(child page 생성 안 함).
> - `notion_push.py` 변환기에 `<details>`→toggle(+nested children), `==..==`→highlight,
>   `>`연속줄→callout 병합 지원 추가. 검증: h1 12/h2 32/h3 3/toggle 7/표 53행/HL 3/리터럴 `**` 0.

## 0. 이 작업이 뭐였나

activation steering 분야를 **하나의 흐름으로 학습·정리**하고 **기술 면접 대비** 자료를 만드는 작업.
산출물 = (1) repo 서베이 문서 + 논문 정독 노트, (2) 논문 PDF 큐레이션(VLA→`references/`, 그 외→
`Activation_steering_basic/`), (3) **Notion 미러** 기입. 방식 = 계층적 agent(주제별 발굴 → 게이트 →
논문별 정독). 사용자 = 본인 연구(VLA pathway/phase conceptor steering) 정리 + 면접 준비.

## 1. 완료 산출물 (검증된 상태)

- **서베이**: `docs/Activation_steering_basic/00_activation_steering_survey.md` (41KB). 구조 =
  한 장 요약 · 핵심 용어 · §1 무엇&왜 · §2 분석(read-out) · §3 steering 방법(write-in) · §4 LLM/VLM ·
  **§5 산업(정직한 현실 점검, 재작성됨)** · §6 VLA/world model · §7 VLA 산업 방향 · 부록 A 면접 치트시트 ·
  부록 B 논문 인덱스(51편).
- **정독 노트 51편**: `docs/Activation_steering_basic/notes/*.md`. 각 노트 = 문제/방법(수식)/결과/
  흐름 위치/우리 프로젝트 연결/면접 Q&A/한계. 서베이의 모든 note 링크는 실제 파일로 resolve됨(검증).
- **PDF**: 기초 33편 `Activation_steering_basic/*.pdf` (%PDF 검증), VLA 19편 `references/*.pdf`.
  - 주의: `references/dr_vla.pdf` = Swann "SAEs Reveal Steerable Features in VLA"(2603.19183). 파일명
    dr_vla = 논문이 공개한 오픈소스 패키지 "Dr. VLA"에서 딴 별칭(오분류 아님). 노트 = `notes/SAE_VLA_pi05.md`.
  - 주의: `MinimizingCollateralDamage_2605.01167.pdf`도 자칭 "COAST" → 우리 COAST(Miao 2605.17144)와
    **동명이인**. 인용 시 arXiv ID로 구분(노트·§5에 경고 있음).
- **Notion**: 페이지 "Activation steer 전반 공부"(id `39063918d42a801ab093f046f22701b4`) 아래
  **11 child page + 상단 인덱스**. 중복 없음(검증). §5 제목 갱신·부록 B 51편 반영됨.
- **README**: `docs/README.md`가 서베이를 reading order에 링크(노트 51편 표기).
- **메모리**: `notion-write-access`, `lit-review-agent-hierarchy` 저장됨.

## 2. ★ 검토 중 발견한 결함 (미수정 — 다음 세션이 고칠 것)

### 결함 A — Notion 렌더링 깨짐 (중요, 우선순위 높음)
서베이 마크다운은 한 문단/불릿이 **여러 물리적 줄로 소프트 줄바꿈**돼 있는데, Notion 변환기
(`_handoff/tools/notion_push.py`의 `md_to_blocks`)가 **줄 단위로 블록**을 만든다. 결과:
- 여러 줄짜리 불릿이 **불릿 1개 + 고아 문단 여러 개**로 쪼개짐.
- `**bold**`가 줄바꿈을 넘어가면 정규식(`\*\*.+?\*\*`, `.`이 개행 불포함)이 매칭 실패 → **리터럴
  `**`가 그대로 노출**. 한 장 요약·§5 등 대부분 페이지에서 발생(실측 확인).
- **FIX**: `md_to_blocks` 진입 전 **continuation line 병합** 전처리 추가 — 어떤 줄이 새 블록 마커
  (`#`, `-`/`*`, `숫자.`, `>`, `|`, ` ``` `, 빈 줄, `---`)로 시작하지 않으면 직전 블록 텍스트에
  공백으로 이어붙인다. 그 뒤 `_handoff/tools/refresh_notion.py`로 **재빌드**하면 해결.

### 결함 B — 서베이 부록 B의 scratchpad 참조 (경미)
`00_activation_steering_survey.md` 마지막(부록 B 끝, ~line 482):
`**스킴/web-only 확장**(정독 대상 아님)은 [../../../scratchpad] 대신 master 인덱스 참고: ...`
→ 세션 임시 경로라 독자에게 무의미. Notion 부록 B 페이지에도 그대로 복제됨.
- **FIX**: 그 문장에서 `[../../../scratchpad] 대신 master 인덱스 참고:` 를 빼고 **스킴/web-only 논문
  목록만** 남긴다(전체 목록·출처는 `_handoff/research_raw/master_index.md`에 보존). repo 수정 후
  Notion 재빌드에 함께 반영.

## 3. 남은 검토 항목 (핸드오프로 중단된 것)

사용자 요청 = "Notion 정리 잘 됐는지 + 논문 선정 적절한지" 검토. 진행 상황:
- **완료 확인**: 서베이 구조·note 링크 무결성(51개 전부 존재)·Notion child page 목록(중복 없음)·
  §5/한장요약/부록B 실제 렌더 내용.
- **미완**: (a) 결함 A/B 수정 및 재빌드, (b) Notion 나머지 페이지(§1~4·§6~7·부록A) 렌더 스팟체크,
  (c) 논문 선정 최종 평가.
- **잠정 평가(선정)**: 8개 주제(정의/분석/방법/왜/LLM·VLM/산업/VLA·WM/산업적용) 모두 커버, 51편으로
  넓고 핵심 논문 포함. 비판 논문(Tan·AxBench·Rogue Scalpel)과 조건부/agentic(CAST·ASA)까지 있어 균형
  양호. 보강 여지: (i) steering **survey/리뷰** 논문 1편이 없음, (ii) "왜 하는가"가 §1·§5에 분산(별도
  묶음 없음) — 필요 판단 시 추가.

## 4. 이어받는 데 필요한 인프라

- **Notion 쓰기**: `.env`의 `NOTION_TOKEN` 유효. 헬퍼 `_handoff/tools/notion_push.py`(stdlib,
  md→Notion blocks; heading 색 h1 red/h2 green/h3 blue). 상대링크는 push 전 plain 변환 필요.
- **재빌드**: `_handoff/tools/refresh_notion.py` — 부모 페이지의 모든 자식 블록 archive 후 서베이에서
  intro + 11 child page를 **순서대로** 재생성. 실행: `set -a; source .env; set +a; python3 <경로>`.
  (child page id는 재빌드마다 바뀌므로 하드코딩 금지. 부모 id만 고정.)
- **PDF 다운로드 패턴**: `_handoff/tools/dl.py`(초기 29편)·`dl2.py`(§5 10편) — arXiv → 폴더, %PDF 검증.
- **원자료(출처 URL 포함)**: `_handoff/research_raw/` — 주제별 발굴 결과(s1~s7), §5 재조사 3갈래
  (A_llm_vlm_deploy·B_imagegen·C_importance_barriers_outlook), 마스터 인덱스, 노트 템플릿. **§5의 모든
  주장 출처 URL은 A/B/C.md에 있음**(서베이엔 요약만).
- ⚠ 위 `_handoff/`는 지금 durable 복사본. 원본은 세션 scratchpad(휘발)라 이 폴더가 유일 백업.

## 5. 다음 세션 시작 프롬프트 (복붙용)

```
docs/Activation_steering_basic/_handoff/HANDOFF.md 를 읽고 이어서 작업해.
먼저 §2 결함 A(Notion 마크다운 소프트 줄바꿈이 리터럴 ** 와 쪼개진 불릿을 만듦)와 결함 B
(부록 B의 scratchpad 참조)를 고쳐. 순서: (1) _handoff/tools/notion_push.py의 md_to_blocks에
continuation-line 병합 전처리 추가, (2) 서베이 부록 B의 scratchpad 문장 수정, (3)
_handoff/tools/refresh_notion.py로 Notion 재빌드, (4) Notion에서 한 장 요약·§5 페이지를 API로
꺼내 리터럴 ** 가 사라지고 불릿이 온전한지 검증. 그 다음 남은 검토(§3): Notion 나머지 페이지
렌더 스팟체크 + 논문 선정 최종 평가. 커밋은 사용자가 지시할 때만.
```

## 6. 지시사항 준수 자가평가 (사용자 참고)

- 하나의 흐름 ✅(7-섹션) · 논문 폴더 분류 ✅(VLA→references, 그 외→basic) · 면접 대비 ✅(치트시트+노트별
  면접 Q&A) · §5 정직성 ✅(과장 지적 후 재조사) · 계층적 agent ✅.
- **미흡**: Notion 렌더 품질(결함 A) — "정리됐다"고 보고했으나 실제 렌더에 `**` 리터럴·불릿 쪼개짐이
  있어 **수정 필요**. 이 부분은 보고가 앞섰음(렌더를 블록수만 보고 내용 렌더는 늦게 확인).
