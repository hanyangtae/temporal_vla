# TTT / Progress Predictor — Legacy (코드 제거됨, 설계만 보존)

예전 연구 방향인 **TTT / VITA progress predictor 기반 실패 루프 탈출**의 보존용 설계 문서.
현재 메인 라인은 latent steering([`../steering/14_pathway_phase_online_steering.md`](../steering/14_pathway_phase_online_steering.md))이며 TTT는 **비메인(legacy)** 이다.

> 정리(2026-06): `src/ttt/` 전체와 phase1 학습/평가 스크립트(`phase1_*`,
> `launch_finetune_ttt.py`, `scripts/eval/phase1_predictor.py`, 관련 `.sh`/테스트), TTT 체크포인트
> 프로파일(`configs/checkpoints/groot_ttt__robocasa_panda_omron.yaml`)을 **삭제했다**. baseline
> GR00T fine-tune 은 TTT 의존을 떼어 `scripts/train/launch_finetune.py` 로 분리했다. 코드 전체는
> git history 참조. 아래는 개념 설계(설계도)만 남긴다.

## 무엇이었나

VLA backbone 재학습 없이 **external progress predictor / TTT token** 으로 실패 루프를 탈출하려던
초기 가설. VITA 식 progress 신호로 "지금 진행 중인지 / 멈췄는지"를 추정해 개입한다.

## 설계 (개념)

2단계 학습:

- **Phase 1 — Progress Predictor**: `TTTModule`(self-supervised inner update) + `ProgressHead` 를
  expert trajectory 에서 학습. label `y_t = t/T`. TTT inner loop 이 temporal context 를 파라미터에
  축적하고 ProgressHead 가 진행률을 예측. 입력 = Eagle pre-LLM 임베딩(dim 2048 = DiT KV dim 고정).
- **Phase 2 — VLA Projector (Δv 기반)**: Phase 1 모듈을 freeze, `VLAProjector` 만 학습.
  `Δv = v(s_{t+1}) − v(s_t)` 로 action 수정 방향 학습. loss = `−Δv · log P(a_expert | s_t, θ_t)`
  → 진전(Δv>0) 구간 expert action 강화, 퇴보(Δv<0) 억제.

주입 방식 (`VLAProjector` modes): (1) `hidden_add` (2) `logit_shift` (3) `film` (4) `token`.

추론: TTT inner loop 이 매 step SSL 로 자체 업데이트 → `ProgressHead` 로 실패 감지(단조증가
이탈) → `VLAProjector` 가 VLA action 출력 수정.

GR00T 통합(삭제됨): `attach_ttt_to_groot` 로 서빙에 선택적 wrap(`Gr00tN1d6WithTTT`), 학습은
upstream `Gr00tN1d6Pipeline._create_model` monkey-patch.

참고 논문: "Learning to (Learn at Test Time)" (Sun et al., 2024), "VITA" (Ziakas & Russo, ICLR 2026).

## 부활 조건

phase-matched steering(메인 method)은 **online phase/progress 신호**를 요구한다. 그 신호 공급원으로
이 progress predictor 가 보조 부품으로 복귀할 수 있다([`../steering/14_pathway_phase_online_steering.md`](../steering/14_pathway_phase_online_steering.md)
"phase를 online에 어떻게 아나" 절). 복귀 시 git history 의 `src/ttt/` 를 복원 기준선으로 쓴다.
