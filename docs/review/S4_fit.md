# S4 — fit (연산자 생성·진단) 스테이지 카드 (2026-08-07)

기계 판독분: [`S4_files.tsv`](S4_files.tsv) · 판정 UI: `python3 scripts/review/ledger_ui.py`

## 규모 — 56파일 14.6k줄, 그러나 라운드 단위로 8무리

steer 트리에서 **fit·연산자·진단만** 골랐다 (실행 러너·큐 = S6, 집계·게이트 = S7 로 유예).
라운드 종결 여부는 D 트랙(RESULTS.md)에서 이미 확정 — 판정은 사실상 무리 단위다.

| 무리 | 파일 | 종결 상태 (RESULTS 근거) | 판정 성격 |
|---|---|---|---|
| 수학 코어 (`src/conceptor`, `operator_config`) | 3 | — 메인 method 토대 | keep 확실 |
| SAE 축 (`src/sae` + `scene_sae`) | 12 | exp5 진행 축 (G1·G2 PASS) | keep 확실 |
| 현행 fit (phase conceptor·setM·placebo) | 5 | 재수집 라운드가 쓸 도구 | keep — 단 좌표 입력 대응 필요 여부 확인 |
| 진단 공용 (비퇴화·NPZ 검사 등) | 4 | 사고 재발 방지 도구 | 개별 판단 |
| exp2 fit | 4 | ★종결 — 위약 대조로 null 확증 | 재현성 보존 vs archive |
| exp3 fit·manifest | 2 | ★종결 — 6-Holm 전부 null ×2 라운드 | 〃 |
| exp4-1 fit·진단 | 7 | 완료 (oracle rescue) — dev 병합됨 | 〃 |
| exp5-4 probe | 5 | ★종결 — seed 암기 판정 | 〃 |
| COAST 재현 (n16) | 2 | 종결 — 비재현 확정 | 〃 |
| patchceil | 3 | 보조실험 (worktree 본류였음) | 〃 |
| **exp4-2 induced** | 11 | **유예** (P0 완료, P1 사용자 결정 대기였음) | ★방향 결정 필요 |

## 판정 축 — 질문 3개

1. **종결 라운드 fit 을 남기나?** (exp2·3·5-4·COAST·patchceil = 23파일 ≈ 5k)
   근거 대립: (a) null 결과의 재현성 — 논문·보고서에서 "재현 가능"을 주장하려면 fit
   코드가 있어야 함. 단 연산자 NPZ·결과 문서는 아카이브에 있고 git 이력으로도 복원
   가능. (b) 구 stem 데이터 전제라 재수집 데이터엔 어차피 못 씀.
   → S1 전례("git rm + 원장에 복원 해시")가 그대로 적용 가능한 영역.
2. **exp4-2(induced) 11파일의 방향.** 교란 수집은 이번 라운드에서 제외됐지만 실험
   자체는 "유예"였지 "폐기"가 아니었다. WA-LQR 선행연구 직결 축 — 살리려면 keep,
   접으면 exp2 와 같은 처분.
3. **현행 fit 5개의 좌표 대응.** fit 스크립트들이 구 stem 경로/manifest 를 읽는지,
   새 좌표 그리드(meta.json·plan)를 읽을 수 있는지 — keep 이어도 '수정' 이 필요할
   수 있다 (재수집 데이터로 fit 하는 시점 전에).

## 참고 — 이미 확정된 사실

- 연산자 저장 규약은 `operator_config.py` 가 강제 (입력 sig 없으면 ValueError) — 배선 완료.
- conceptor 수학은 exp 라운드들이 null 이어도 **메인 method 토대로 유지** 결정된 상태.
- exp5-3 의 fit(within-scene setM)이 현행 표준 — β=1.0 파괴 사고 후 per-scene setpoint 확립.
- steer 루트의 `plan/run_instruction_steering_eval.py` + `.sh` 러너들은 S6 에서 판정
  (이번 S1 적용으로 collect_instruction_fixed... 가 사라져 이미 깨진 것 포함).
