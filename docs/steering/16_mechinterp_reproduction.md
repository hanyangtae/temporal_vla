# 16 — Mechanistic Interpretability Steering 재현(Phase A) + 우리 연구와의 관계

> Häon et al. CoRL 2025 *"Mechanistic Interpretability for Steering VLAs"*(arXiv 2509.00328)
> 재현 기록과 우리 메인 연구(`14_pathway_phase_online_steering.md`)에 대한 의미. (2026-06)

## 무엇을 했나
- **Phase A**(저자 공개 코드, vanilla OpenVLA-7B + LIBERO-10 sim) 재현 완료.
- 재현 구동·분석 스크립트: `scripts/analysis/mech_interp/`(README 에 실행법·환경 함정).
- 저자 코드는 third-party 라 vendoring 안 함 → 별도 클론 `~/pkt_ws/mechanistic-steering-vlas`,
  env `openvla-interp`. **우리 서빙/eval 스택으로의 메서드 이식(Phase B)은 미구현**(아래).

## 결과
**해석(logit-lens) ✅ 재현**
- 백본 FFN value vector 가 의미 토큰으로 디코드(예: previous/already/old/earlier 같은 일관 클러스터).
- action-token 이 후반 레이어에 집중: L0~18 ~39% → L31 86%(논문 Fig 2b).
- 우리 CPU 투영이 저자 배포 `up_10` 클러스터와 10/10 일치 → 투영 신뢰.

**스티어링(override) — fast/slow, coef=6, LIBERO-10, 5 trial(n=50/조건)**

| 비교 방식 | fast/slow | 비고 |
|---|---|---|
| 전체 속도(full episode) | +26.3% | 길이 confound 미통제 |
| 길이통제(task별 최단길이 truncate) | +15.7% | 길이통제 시 효과 ~반감 |
| 논문 방식(fast vs slow, 10 task paired) | +37.6% | paired t p=0.029, dz=0.82; ratio-평균이 부풀림 |

- fast vs slow 는 **통계적으로 유의**(8/10 task) → 논문 방향성 재현.
- **단 fast·slow 둘 다 baseline 보다 느림**(fast −19%, slow −36%). "fast"는 "baseline 보다
  빠름"이 아니라 "slow 보다 덜 느림" — 논문 로봇 섹션의 "high/fast resembled baseline" 과 합치.
- **비교 방식이 결론 크기를 좌우**: 논문 방식(+38%)이 가장 큼(per-task ratio 평균·길이 미통제),
  길이통제하면 절반(+16%). 논문은 효과를 크게 보이는 프레이밍 채택(+coef sweep 평균으로 p<0.001).

## 우리 메인 연구(`14_`)와의 관계
- **축 대비**: 논문 = vocab-의미·단일 neuron override·정적·레이어국소·백본LM(goal). 우리 =
  결과(succ/fail)유래·다차원 conceptor(`h·Mᵀ`)·rollout-phase·online 라우팅·VL+DiT 분리.
  논문은 우리 사다리의 **바닥 baseline**(정적 의미 steering)이자 novelty 경계의 대조군.
- **미탐사 pathway 개통**: `14_` 가 "아직 tap 안 함, goal 후보"로 남긴 **백본 LM(Eagle/PaliGemma)**
  을 이 방법(logit-lens + 백본 FFN override)이 정확히 연다.
- **작동하는 positive control**: COAST positive control 은 실패(ΔSR≈0)였으나, 이 방법은 모션을
  실제로 바꿈을 입증(개입이 먹히긴 함) → 우리 방법이 ΔSR 로 이겨야 할 baseline 제공.
- **길이 confound 재확인**: OpenVLA 에서도 길이통제 시 효과 ~반감 → `01_`/seen18 의 "length-fair
  비교 필수" 를 외부 모델에서도 검증. 동시에 "유의 신호는 실재"라 다차원·phase 방법의 여지 시사.
- **fine-tuning↔steerability**: vocab 엔 의미가 보여도 fine-tuning 이 그 방향-행동 정렬을 끊을 수
  있음(논문 Limitations). 우리 RoboCasa-finetuned ckpt 의 VL-pathway 방향 접근성 점검 필요.

## Phase B (미구현, 다음 단계)
- pi0.5(PaliGemma 백본) → RoboCasa 에 logit-lens + `down_proj` override 이식.
  - Stage1(싸게): 백본 FFN value-projection → 의미 neuron 확인 + 클러스터 + 주입 수치검증.
  - Stage2(비쌈): `ValueVectorSteering` hook(백본 down_proj) + `lerobot.py` 배선 +
    `robocasa_eval.py` per-step eef 로깅 + baseline vs fast/slow 변위·SR.
  - flow-matching = 2-시스템(백본+별도 action head)이라 **백본 의미 override 가 모션으로
    전이되는지가 핵심 질문**(우리 VL→motor 가설의 feasibility 측정). 음성도 유효 결과.
- 검증된 모듈 경로·caveat(absolute action→np.diff, Fig2b 해당없음, single-shot conditioning)는
  plan 파일 참조.
