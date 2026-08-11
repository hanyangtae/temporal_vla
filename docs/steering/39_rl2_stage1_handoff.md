# 39. RL2-VLA 이식 Stage 1 핸드오프 (다음 세션용)

2026-08-11 작성. **Stage 1(SIMPLER 재현) 완결** 시점의 상태 스냅샷.
결과·판정의 단일 출처 = [`38_rl2_simpler_repro.md`](38_rl2_simpler_repro.md),
논문 이해 = [`../references/reading_notes/rl2_vla_adaptive_steering.md`](../references/reading_notes/rl2_vla_adaptive_steering.md),
경쟁 지형 = memory `rl2-vla-competitor-analysis` + `vla-tts-adoption-survey`.

## 한 문단 요약

RL2-VLA(arXiv 2607.26991, 우리 "언제 개입" 축의 baseline)를 fork·서브모듈로 통합하고
SIMPLER에서 논문 프로토콜 풀사이즈(3 seed × α top-3 × 4 arm) 재현 완료. **baseline 3 arm은
공개물만으로 재현**(Rephrase 45.3=논문 45.3 소수점 일치), **adaptive는 공개 SAFE ckpt로는
−7.0pp 미재현** → 3-agent 원인조사(코드 diff·게이트 실측·ckpt 출처)로 원인을 "번들 SAFE의
플랫폼 점수분포 미스매치"로 국소화 → 저자 README 권고대로 **우리 rollout 1,200판 수집 →
SAFE 재학습 → α 재산출 → adaptive 재평가 = 52.7로 논문(53.8±3.2) 오차범위 진입, 복원 확정**.
전 과정 5,400 에피소드, 에러 0.

## 최종 수치 (전부 3 seed, OOD 환경 suite)

| arm | 우리 | 논문 |
|---|---|---|
| Vanilla | 38.0 | 36.0 |
| Rephrase | 45.3 | 45.3 |
| Compose-Always | 47.7 | 46.5±1.8 |
| Adaptive (번들 SAFE) | 48.2 | — |
| **Adaptive (재학습 SAFE)** | **52.7** (α고정 시 49.5) | **53.8±3.2** |

핵심 단서: α 민감도 ±5pp(Δα=0.05), 이득 상당분이 (seed×task) 셀별 α 사후선택 프로토콜에
실림. 상세 판정·불일치쌍 분석(합성이 망친 판 89/구한 판 103)·SAFE 지표의 길이 confound
오염(0.756→공정 0.54~0.59) 전부 38 문서에.

## 인프라 지도 (재실행에 필요한 전부)

- **worktree**: `.claude/worktrees/rl2-vla-port`, 브랜치 `feat/rl2-vla-port` (dev 분기).
  메인 트리는 건드리지 않음. 서브모듈 `RL2-VLA/` = fork `hanyangtae/RL2-VLA` (+내부 SAFE·qam).
- **conda env**: `rl2` (py3.10, torch 2.5.1+cu121). 설치 스크립트의 sudo 줄만 제외하고 실행함.
- **ckpt**: pi0=HF `juexzz/INTACT-pi0-finetune-bridge`(캐시됨), QAM/CoVer=서브모듈 내
  지정 경로에 다운로드됨, **재학습 SAFE = `RL2-VLA/third_party/SAFE/logs/open_pizero-bridge-lstm-ours_cpTrue/20260807/123421/`**
  (combined CP; per-task는 `..._cpFalse/20260807/123255`).
- **러너** (`scripts/rl2_vla/stage1_simpler/`):
  - `run_arm.sh <arm> <gpu> [suite] [seed] [trials] [alpha]` — arm 하나. env `SAFE_DIR_OVERRIDE`
    (재학습 SAFE 경로), `LANE_TAG` 지원.
  - `run_full_repro.sh "<gpus>"` — 풀 재현 큐. `collect_safe_rollouts.sh` — SAFE 데이터 수집.
  - `rerun_adaptive_ours.sh` — 재학습 SAFE adaptive 재평가. `aggregate.py` — 집계
    (⚠ "oracle" 라벨 모드가 실제 저자 프로토콜 = (seed×task) 셀별 α max).
  - `compute_alpha_heuristic.py` — α top-3 재산출 (저자 함수 import, bit 검증됨).
    산출 JSON = `RL2-VLA/experiments/rl2_cp_alphas_combined_ours.json`.
- **로그·데이터**: eval 로그 = `RL2-VLA/experiments/` (stage1b_*·full_*·ours_adaptive_*),
  수집 rollout 9.6GB = `RL2-VLA/safe_rollouts/` (latent 포함 pkl 1,200개 — Bridge 세계
  분석용으로 재사용 가치 있음). 진행 중인 백그라운드 작업 **없음**, GPU 전부 반납됨.

## 노션 (교수님 공유용 완성본)

페이지 id `3b363918d42a80be88b6ead611406ccd` ("RL²(robocasa, gr00t)").
구조: 논문소개·방법 토글(verifier 학습법 포함) → h1 결과 아래 설계(OOD 구성·성공영상
4쌍)·Method(rephrase/compose 상세 토글)·결과(우리 최종표+논문 Fig8표)·1차 판정·원인
규명·시사점. 전부 개조식으로 통일됨. ⚠ 사용자가 직접 편집 중인 페이지 — 수정 전 반드시
현재 블록을 되읽고, 사용자 작성 블록(recap 등) 보존.

## 미결·다음 단계

1. **Stage 2 (RoboCasa+pi0.5 이식) 게이트 — 사용자 결정 대기**: ① verifier 전략
   (CoVer는 Bridge 전용·RoboCasa용 부재 — 재학습 vs verifier-free vs oracle)
   ② QAM 학습 데이터(로컬 atomic 데모 ~1천ep vs 우리 rollout vs RoboCasa365 다운로드
   — QAM ckpt는 RoboCasa용 공개본 없음, 재학습 필수 확정) ③ eval task set.
   배선 지점은 계획 파일(`~/.claude/plans/iridescent-tickling-sunrise.md`)과 memory에 확정돼
   있음 (pi0.5 denoising 10step의 `sample_actions` v_t, N1.5는 4step ⚠).
2. **Stage 2 설계 교훈 (확정)**: 게이트 threshold는 사후선택 없는 고정값 primary·스윕은
   민감도 곡선; scene 통제(RoboCasa는 에피소드마다 layout·style 다름 — SIMPLER와 정반대);
   fit/eval seed 분리.
3. **선택 과제(미착수)**: 재학습 SAFE의 발동 타이밍 실측(번들 대비 빨라졌나), latch 실험,
   IID suite 재현, phase-milestone 보상 shaping(우리 라벨러→QAM 보상 채널 — 개선 기여 후보).
4. **git**: 브랜치에 커밋 6개(서브모듈·러너·docs 38). PR은 dev로, 아직 안 열음.
   `RL2-VLA/experiments`·`safe_rollouts`는 untracked(대용량 — 커밋 금지).

## 함정 (재실행 시)

- 저자 스크립트 `INFERENCE_ROOT` 경로 낡음 → PYTHONPATH 직접 지정 (run_arm.sh가 처리).
- SAFE 학습 `data_path_prefix`는 **trailing slash 필수** (문자열 연결 버그).
- `WANDB_MODE=offline` 필수. SIMPLER 성공 판정 순간 에피소드 즉시 종료 → 성공 영상도
  마지막 프레임이 어색할 수 있음(라벨은 env 판정이 정답). 일부 mp4는 1프레임 손상
  (spoong ep30) — 프레임 수 확인 후 사용.
- GPU: 빈 GPU만, 장시간 run은 setsid nohup + sentinel + 행수 검증. 판당 rephrase 1.3분/
  adaptive 1.9~2.9분, VRAM 13.3GB/lane.
