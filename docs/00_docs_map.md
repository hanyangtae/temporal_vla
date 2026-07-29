# 문서 지도

**이 파일은 경로 참조 전용이다.** "어떤 문서가 어디에 있는지"만 답한다.
프로젝트가 무엇이고 현재 연구 상태가 어떤지는 루트 [`README.md`](../README.md)를 본다.

## 루트 runbook (읽기 순서)

1. [01 Serving Interface](01_serving_interface.md) — 통일 HTTP API 단일 출처. endpoint(`/act`,
   `/act_with_features`, `/reset`, `/health`), sub-key 네임스페이스, 모델 × 벤치마크 호환 매트릭스.
   모델/벤치/체크포인트 작업의 출발점.
2. [02 Docker Guide](02_docker_guide.md) — 컨테이너 구성·기동·VNC/X11·troubleshooting.
3. [03 Adding Checkpoint](03_adding_checkpoint.md) — 새 체크포인트를 profile/serve/eval에 붙이는 체크리스트.
4. [CONTRIBUTING](CONTRIBUTING.md) — git/PR 워크플로우.

부가 runbook:

- [Cache Paths](cache_paths.md) — 체크포인트·데이터셋의 repo 밖 cache 위치와 경로 참조 규칙
  (`path_setup.py` / `cache_env.sh`).
- [A100 Offload Plan](a100_offload_plan.md) — srv50(A100) 이관 계획.
- [seen18 SAFE Detector Verification](seen18_safe_detector_verification.md)

## 연구 — Latent Steering (메인 라인)

[`steering/`](steering/README.md) 아래. 라운드별 계획·결과가 번호 prefix로 쌓여 있다.

- **문제 설정 단일 출처**: [14 Pathway+Phase Online Steering](steering/14_pathway_phase_online_steering.md)
- **연구 구조**: [15 Research Structure](steering/15_research_structure.md) — RQ1~4 / 가설 C1~C4
- **표현 분석**: [01 seen18 Latent Analysis](steering/01_seen18_latent_analysis.md) (길이 confound 통제 전제),
  [08 Pathway Separation](steering/08_pathway_separation_analysis.md) ⚠ §2.1 반증됨,
  [11 Instruction Confound](steering/11_instruction_confound.md)
- **방법론 후보**: [07 Steering Methods Survey](steering/07_steering_methods_survey.md)
- **라운드 결과**: exp2 = `23_`, exp3 = `21_`·`22_`, exp4 = `28_`, exp5 = `31_`·`32_`·`33_`·`35_`·`36_`·`37_`

> ⚠ `steering/README.md`의 "현재 결론"과 Reading Order는 06-23에 멈춰 있어 exp2~exp5가
> 반영되지 않았다. 레포 검토(아래) D 트랙에서 연구 연대기 인덱스로 교체 예정.

한 화면 요약: [GR00T Latent Steering Explorer](groot/00_groot_steering_explorer.html) — runtime flow,
conceptor/hidden-state steering 수식, code artifact map을 함께 보는 self-contained HTML.

## GR00T 실행 절차

[`groot/`](groot/README.md) 아래. README에 reading order가 있다.

- **먼저 볼 것**: [GR00T Flow Map](groot/00_groot_flow_map.md) — LeRobot/native, RoboCasa365,
  ZMQ/HTTP entry point의 call chain. [Data Catalog](groot/00_groot_data_catalog.md) — rollout·activation이
  로컬/원격(승준 HDD) 어디에 있는지.
- **N1.6**: `n16_01_finetune` → `n16_02_eval` → `n16_03_safe_overview` → `n16_04_safe_collection`
  → `n16_05_safe_env_reproduction` → `n16_06_safe_inference_semantics` → `n16_07_safe_detector_report`
  → `n16_09_safe_parity` → `n16_11_http_act_changes`
- **N1.5**: `n15_01_finetune` → `n15_02_eval` → `n15_03_lerobot_robocasa365`

N1.5와 N1.6의 DiT token layout은 대칭이 아니다. feature shape 비교 전에
[`groot/README.md`](groot/README.md#n15-n16-feature-contract)의 quick reference를 먼저 확인한다
(N1.6 full residual `T=51 = state(1)+action(50)`, N1.5 aligned residual `T=49 = state(1)+future(32)+action(16)`).

## 레포 검토·정리 (진행 중)

- [설계](superpowers/specs/2026-07-28-repo-review-design.md) — 파이프라인 축 순회 계획
- [`review/`](review/) — 스테이지 카드 `S<N>_*.md` + 판정 원장 `LEDGER.tsv`

## 참조

| 위치 | 내용 |
|---|---|
| [`benchmarks/`](benchmarks/) | RoboCasa task 이름 매핑, env 결정성·재현 절차 |
| [`references/`](references/) | 논문 PDF + 정독 노트 (`reading_notes/`) |
| [`related_work/`](related_work/) | 우리 방법과의 차이 정리 (reviewer 대응용) |
| [`Activation_steering_basic/`](Activation_steering_basic/00_activation_steering_survey.md) | activation steering 서베이 52편 + 노트 |
| [`adr/`](adr/) | 장기 결정 기록 |
| [`collab/`](collab/) | Codex 협업·Gate 리뷰 기록 |
| [`onboarding/`](onboarding/00_intern_curriculum.md) | 인턴 커리큘럼 |
| [`insight/`](insight/) | INSIGHT VLM primitive-segmentation 파일럿 |
| [`ttt/`](ttt/README.md) | 구 TTA/progress predictor (메인 아님, 보존) |
| [`_legacy/`](_legacy/) | 폐기된 repo-wide snapshot |

## 배치 규칙

- **Project-wide runbook** — `docs/` 루트, 번호 prefix(`01_`, `02_`, `03_`)로 읽기 순서 표현.
- **GR00T 실행 절차** — `groot/`, 번호 prefix(`n16_NN_`, `n15_NN_`).
- **Latent steering 라운드 기록** — `steering/`, 번호 prefix(`01_` ~ `NN_`).
- **벤치마크 reference** — `benchmarks/`. **장기 결정** — `adr/`.
- **논문 PDF/외부 reference** — `references/`. **해석 노트** — `related_work/`.
- **Legacy/폐기 문서** — 해당 scope의 `_legacy/`. scope가 애매한 repo-wide snapshot은 `docs/_legacy/`.
  파일명 prefix `legacy_*` 는 쓰지 않는다.
