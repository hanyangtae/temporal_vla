---
name: weekly-report
description: Notion Weekly Report 주간 리포트 작성. 가장 최근 리포트의 TODO·git log·docs·agent memory·outputs 실험 결과를 수집해 마크다운 draft를 만들고, 사용자 승인 후 Notion 해당 주차 toggle heading에 업로드한다. "주간 보고", "위클리 리포트", "weekly report 써줘" 등에 사용.
---

# Weekly Report 작성

> **스타일 단일출처: `notion-polish` skill** (user-level, `~/.claude/skills/notion-polish`).
> 리포트 본문 작성·업로드 시 그 컨벤션을 따른다 — h1 빨강/h2 초록/h3 파랑 배경(글씨 검정),
> callout 한 줄 요약 → 짧은 불릿 3~5개, 결과는 테이블(해석은 아래 불릿), 장황한 문단 금지,
> 수식 $ 금지(plain/unicode), infra 세부(seed·포트·GPU) 제외, 사용자 작성 블록 보존,
> 쓰기 후 API로 되읽어 검증.

Notion `Weekly Report` 페이지에 이번 주 연구 리포트를 작성하는 워크플로우.
Notion 통신은 전부 `scripts/utils/notion_weekly_report.py` CLI 로 한다 (토큰은 `.env` 의
`NOTION_TOKEN`, 하드코딩 금지).

## 페이지 구조

`Weekly Report` → 연도 child_page (`2026`) → 월 child_page (`6월`) → 주차별 **toggleable
heading_1** → 내용 블록은 heading 의 children.

- **주차 이름은 보고 당일이 그 월의 몇째주인지**로 결정 (`N째주`). 예: 6/11 보고 → `2째주`.
- **보고를 건너뛴 주가 흔함**. "지난주 리포트"가 아니라 `latest-report` 가 찾아주는
  **가장 최근에 내용이 있는 주차**가 Recap 소스다.

## 절차

### 1. 주차 판정 + 대상 heading 확보

```bash
python3 scripts/utils/notion_weekly_report.py resolve --year <YYYY> --month <M> --week <N> --create
```

`<N>` = 오늘(KST)이 그 월의 몇째주인지 (1일이 포함된 주가 1째주, 일요일 시작 기준.
애매하면 사용자에게 확인). 기존 heading 이 있으면 재사용된다 (`2째주`/`2주차` 변형 모두 인식).

### 2. 정보 수집

- **최근 리포트**: `latest-report` → `read-week --block-id <id>` 로 덤프.
  마지막 `# TODO` 섹션이 이번주 Recap 의 소재. **단, TODO 는 참고일 뿐**: 전부 다룰
  필요 없고, TODO 에 없던 작업이 본문에 들어가는 것도 당연. 본문은 **실제 한 일** 기준.
- **git log**: `git log --since=<최근 리포트 날짜> --oneline` + 그 기간 변경된 `docs/`
  문서 (핸드오프·분석 문서 우선).
- **agent memory**: `MEMORY.md` 와 관련 메모리 파일 (연구 방향 전환, 새 발견).
- **outputs/**: `outputs/eval/` 등 최근 실험 산출물에서 SR 수치·결과 (표 소재).
- 부족한 내용은 사용자에게 그때그때 질문.

### 3. Draft 작성

`outputs/weekly_report/YYYY-MM-wN.md` 에 마크다운 초안 작성.

**구조**: 시작 `# Recap` 과 끝 `# TODO` 는 고정. **중간 섹션은 그 주 실제 작업에 맞춰
매번 달라진다** — 구성을 먼저 사용자와 합의하고 쓰는 편이 좋다. 주제 heading_1 아래
`## <모델> + <벤치마크>`, `### case N` 세분화는 내용에 따라 선택.

**문체**:
- 짧은 한국어 연구노트체 ("~함", "~확인", 명사형 종결). 과장 없이 사실 위주.
- 수식은 plain/unicode 문자 (LaTeX `$` 금지).
- 근거 논문·부가 설명은 toggle 로, 핵심 결론은 본문 paragraph 로.

**upload 가 인식하는 마크다운** (이외 문법은 plain text 로 들어가니 쓰지 말 것):
- `#`/`##`/`###` heading — 자동으로 색 적용 (h1=red, h2=green, h3=blue, 사용자 선호)
- `- ` bullet, `1. ` numbered list, `---` divider, 마크다운 표
- `> toggle: 제목` + 이어지는 들여쓰기 줄 = toggle 블록
- `[이미지: 설명 — <로컬 png 경로>]` = 경로가 실재하면 **File Upload API 로 실제 이미지 업로드**
  (caption = 설명). 경로가 없거나 파일이 없으면 회색 placeholder (사용자 수동 삽입).
  기존 분석 산출물(plot)을 우선 재사용하고, 새 figure 생성은 기존에 없는 경우만.
- 굵게/기울임/링크 등 인라인 서식은 미지원

### 4. 승인 루프

Draft 를 사용자에게 보여주고 수정 반영. **명시적 승인 전에 업로드 금지.**

### 5. 업로드

```bash
python3 scripts/utils/notion_weekly_report.py upload --block-id <week_block_id> --draft outputs/weekly_report/YYYY-MM-wN.md
```

업로드 후 보고: Notion 페이지 URL (`https://app.notion.com/p/robots-oh/Weekly-Report-1ac63918d42a804eb403e37471a837c6`),
append 된 블록 수, 그리고 **수동 삽입이 필요한 이미지/비디오 placeholder 목록**.
