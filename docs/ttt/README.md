# TTT / Progress Predictor — Legacy (demoted)

예전 연구 방향인 **TTT / VITA progress predictor 기반 실패 루프 탈출**의 보존용 단일 문서.
현재 메인 라인은 latent steering([`../steering/RESEARCH_DIRECTION.md`](../steering/RESEARCH_DIRECTION.md))이며 TTT는 **비메인(legacy)** 이다.

> 정리(2026-06): 상세 설계 문서(`ttt_pipeline.md`, `progress_predictor.md`)와 상태 스냅샷
> (`2026_04_01_status.md`, `_legacy/`)은 제거했다. 전체 내용은 git history 참조. 전용 agent
> (`ttt-module-manager`)와 그 agent-memory 도 함께 제거했다.

## 무엇이었나

VLA backbone 재학습 없이 **external progress predictor / TTT token** 으로 실패 루프를 탈출하려던
초기 가설. VITA 식 progress 신호로 "지금 진행 중인지 / 멈췄는지"를 추정해 개입한다.

## 코드 (보존)

`src/ttt/` 가 핵심 모듈로 남아 있다:
- `ttt_module.py` (TTTModule, self-supervised inner update), `progress_head.py` (ProgressHead),
  `vla_projector.py` (4 injection modes: hidden_add / logit_shift / film / token),
  `predictor.py` (ProgressPredictor, Phase 1 학습 → Phase 2 freeze), `losses.py`.
- `integrations/groot_wrapper.py` 의 `attach_ttt_to_groot` 로 GR00T 서빙에 **선택적**으로 붙는다
  (`src/policies/groot/core/service.py` 의 `_attach_ttt_if_requested`, model-spec 로 on/off).
- 학습/평가: `scripts/train/phase1_*`, `scripts/train/launch_finetune_ttt.py`,
  `scripts/eval/phase1_predictor.py`.

## 부활 조건

phase-matched steering(메인 method)은 **online phase/progress 신호**를 요구한다. 그 신호 공급원으로
이 progress predictor 가 보조 부품으로 복귀할 수 있다 ([`../steering/RESEARCH_DIRECTION.md`](../steering/RESEARCH_DIRECTION.md)
"phase를 online에 어떻게 아나" 절). 그 전까지는 손대지 않는다.
