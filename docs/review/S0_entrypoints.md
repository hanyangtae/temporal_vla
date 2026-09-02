# S0 — 진입점·문서 지도

> 스테이지 카드. **판정 열은 사용자가 채운다.** 작성일 2026-07-28 · 기준 커밋 `92729ee`

## 1. 대상 파일

| 파일 | 줄 | 최종수정 | 역할 | 판정 |
|---|---|---|---|---|
| `README.md` | 280 | 06-23 | 레포 최상위 소개·구조·빠른 시작 | |
| `CLAUDE.md` | 192 | 07-10 | Claude Code 세션 지침 (연구방향·평가표준·경로규칙) | |
| `AGENTS.md` | 8 | 05-26 | Codex 계열 entrypoint (stub) | |
| `.agents/agent_spec.md` | 255 | 06-09 | agent 운영 규칙 단일 출처 | |
| `docs/README.md` | 163 | 07-06 | docs 전체 인덱스·읽기 순서 | |
| `docs/HANDOFF_current.md` | 161 | 07-22(내용 07-10) | 새 세션 진입점 | |
| `docs/steering/README.md` | 89 | 06-23 | 메인 연구 라인 인덱스 | |

## 2. 기계 검사 결과

### 2.1 링크 — 문제 없음

진입점 6개 문서의 상대 링크 **63개 전부 해석됨**(깨진 링크 0). 파일 이동·삭제로 인한
링크 부패는 없다.

### 2.2 도달 가능성 — 심각

`docs/README.md`에서 링크를 따라갔을 때:

```
docs/ 전체 md 165개 · 도달 가능 94개 · 도달 불가 71개 (43%)
```

도달 불가 71개의 분포:

| 개수 | 위치 | 성격 |
|---|---|---|
| 32 | `docs/steering/` | **exp2~exp5 전 라운드 문서 — 이번 정리의 핵심 문제** |
| 11 | `Activation_steering_basic/_handoff/research_raw/` | 서베이 원자료. 의도된 미인덱싱으로 보임 |
| 8 | `docs/collab_codex/` | Codex 협업 기록 |
| 7 | `references/reading_notes/` | 논문 정독 노트 |
| 3+1 | `docs/superpowers/` | specs·plans (본 설계 문서 포함) |
| 나머지 | `insight/`, `onboarding/`, `steering/etc/` 등 | |

**`docs/steering/` 도달 불가 32개 = 16번부터 37번까지 전부.** 즉 exp2·exp3·exp4·exp5
네 라운드가 만든 모든 문서가 인덱스에서 보이지 않는다.

### 2.3 내용 현행성 — 심각

진입점 문서에서 라운드 키워드 출현 횟수:

| 문서 | exp2 | exp3 | exp4 | exp5 | SAE |
|---|---|---|---|---|---|
| `README.md` | 0 | 0 | 0 | 0 | 0 |
| `CLAUDE.md` | 0 | 0 | 0 | 0 | 0 |
| `docs/README.md` | 0 | 0 | 0 | 0 | 0 |
| `docs/steering/README.md` | 0 | 0 | 0 | 0 | 1 |
| `docs/HANDOFF_current.md` | 1 | 0 | 0 | 0 | 3 |

`docs/README.md`가 인덱싱하지 않는 하위 디렉토리: `collab_codex/`(13), `insight/`(2),
`onboarding/`(1), `superpowers/`(2).

## 3. 발견된 함정

1. **`docs/steering/README.md`의 "현재 결론"이 06-23 시점에 멈춰 있다.** 그 이후 확정된
   판정들 — exp2 raw 대조 conceptor 종결, exp3 6-Holm 전부 null·COAST 비재현 재확정,
   exp5 read≠write, exp5-4 seed 암기 귀인 — 이 하나도 반영돼 있지 않다. 이 README만 읽으면
   "다음 실험 축: fixed-instruction confound 제거 → VL conceptor fit → ΔSR 재측정"이
   다음 할 일로 보이는데, 실제로는 그 경로가 여러 라운드에 걸쳐 종결됐다.

2. **`docs/HANDOFF_current.md`의 파일 날짜(07-22)와 내용 날짜(07-10)가 다르다.** 07-22
   일괄 커밋에 휩쓸린 것으로, 내용은 exp2 매트릭스 완료 시점에서 멈춰 있다. "새 세션 진입점"을
   자처하는 문서가 두 라운드 뒤처져 있다.

3. **`docs/steering/README.md`의 Reading Order가 8개 문서만 나열한다** (01·05·07·08·11·11·13·15).
   43개 중 8개. 나머지 35개는 번호만 있고 순서상 위치가 없다.

4. **`AGENTS.md`가 8줄 stub이고 05-26 이후 미수정.** `.agents/agent_spec.md`(06-09)로
   포워딩만 한다. CLAUDE.md는 "AGENTS.md는 참고 문맥으로 사용"이라고 규정하므로 의도된
   구조일 수 있으나, 실질 내용이 없다.

5. **`docs/README.md`가 `steering/11_phase4_n15_instruction_fixed_plan.md`(1421줄)를
   현행 계획으로 가리킨다.** 해당 라운드는 종결됐다(N1.5 COAST 재현 실패로 중단).

## 4. 진입점 실행 검증

S0은 실행 코드가 없다. 대신 문서가 주장하는 진입점이 실제 존재하는지는 링크 검사(2.1)로
확인됐고, 그 진입점이 **현재 유효한 절차인지**는 해당 코드 스테이지(S1~S9)에서 확인한다.

## 5. 판정이 필요한 것

판정은 UI(`python3 scripts/review/ledger_ui.py`)에서 하고 `LEDGER.tsv`에 기록된다.
판정 5종의 뜻은 UI 상단 범례 참조. **사용자는 파일을 직접 고치지 않는다** — 판정과
사유만 남기면 스테이지 끝에서 에이전트가 적용한다. 사유칸이 작업 지시서다.

| 파일 | 예상 판정 | 사유칸에 들어갈 내용 |
|---|---|---|
| `docs/steering/README.md` | `수정` | "현재 결론을 exp2~5 판정으로 교체 + Reading Order에 16~37 추가" — **D 트랙 후 실행** |
| `docs/HANDOFF_current.md` | `수정` 또는 `archive` | 연대기 인덱스를 새로 만들면 이 문서의 역할이 없어질 수 있음 — 실제 판단 필요 |
| `docs/README.md` | `수정` | "미인덱싱 4개(collab/insight/onboarding/superpowers) 추가 + 종결된 11_phase4 참조 정리" |
| `AGENTS.md` | `keep` 또는 `archive` | 의도된 stub인지 |
| `README.md` | 정독 후 `keep`/`수정` | 기계 판정 불가 |
| `CLAUDE.md` | 정독 후 `keep`/`수정` | 기계 판정 불가. **"연구 방향" 절을 exp2~5 결과와 대조** |
| `.agents/agent_spec.md` | 정독 후 `keep`/`수정` | 기계 판정 불가 |

뒤 세 개는 현행성을 기계적으로 판정할 수 없어 사용자 정독이 필요하다.

## 6. 권고 순서 (판정 대기)

1·2·3은 **D 트랙 완료 후**가 자연스럽다. 결과 문서 12개를 읽고 연대기를 만든 뒤라야
"현재 결론"에 무엇을 쓸지가 정해지기 때문이다. 지금 갱신하면 D에서 다시 고쳐야 한다.

4·5·6은 지금 처리 가능한 국소 판정이다.

7~9는 사용자 정독 후 판정.
