# Mechanistic Interpretability for Steering — 재현 (Phase A)

Häon et al. CoRL 2025 *"Mechanistic Interpretability for Steering VLAs"*
(arXiv 2509.00328) 의 **Phase A 재현**(OpenVLA + LIBERO sim) 스크립트·분석 기록.

논문 method: FFN `down_proj` 의 열(=value vector)을 LM head 로 vocab 에 투영(logit-lens)
→ top token 으로 의미("fast"/"slow"/"up") 부여 → 같은 의미 neuron 집합 S 의 활성을 추론 시
스칼라 α 로 **override** 하여 행동을 steer (백본 재학습 없음). 논문 본 PDF:
`docs/references/Mechanistic Interpretability for Steering.pdf`. 연구 맥락·결과 해석:
`docs/steering/16_mechinterp_reproduction.md`.

## 코드가 두 곳에 있는 이유
- **이 디렉토리** = 우리가 작성한 재현 구동·해석·시각화 스크립트 + 결과 아티팩트.
- **저자 공개 레포**(third-party: openvla/openpi/libero_experiments 전체)는 vendoring 하지
  않고 **별도 클론** 에서 돌린다: `~/pkt_ws/mechanistic-steering-vlas` (git clone).
- 메서드를 **우리 서빙/eval 스택에 이식**하는 것은 Phase B(pi0.5/GR00T + RoboCasa)이며
  아직 미구현 — 설계는 plan 파일/`docs/steering/16` 참조.

## 환경 (재현 시 함정 포함)
- conda env `openvla-interp` (저자 `setup/openvla/environment.openvla.yml` 기반, 단:
  **flash-attn 제거**(nvcc 없음 → sdpa 폴백), **bitsandbytes 넣지 말 것**(torch 를
  2.12+cu13 으로 끌어올려 GPU 죽음), **mujoco==3.8.1 핀**(3.10 은 robosuite 1.4.0 의
  `mj_fullM` 비호환)).
- **bf16 OpenVLA-7B 가 A4000 16GB 에 ~14.6GB 로 딱 맞음** → 8-bit 불필요. 단 value
  projection 은 model+vectors 동시 GPU 면 OOM → **CPU fp32** 로 투영(safetensors 에서
  lm_head 만 로드).

## 실행 순서 (저자 클론 + openvla-interp 활성 가정)
1. `value_projection.py` — 체크포인트 down_proj 열을 lm_head 로 투영 → `top_tokens_output.txt`.
2. `build_clusters.py` — top_tokens 에서 fast/slow/up 클러스터(flat_idx) 구성 → dict yaml.
3. `verify_interpretability.py` — 의미 value vector·action-token 후반집중(Fig 2b)·저자
   클러스터 일치 확인.
4. `verify_injection.py` — 실제 down_proj 가중치로 override hook 이 선택 neuron 만 정확히
   α 로 바꾸는지 수치 검증(3 check).
5. `run_eval_compare.sh` — LIBERO-10 baseline/fast/slow 롤아웃(영상·actions.json).
6. `analyze_displacement.py` — 3-way 변위 비교(전체속도 / 길이통제 / 논문 paired t-test).
7. `visualize.py` — Fig 2b·클러스터·변위·주입검증 플롯.

**경로 주의**: 스크립트는 저자 클론(`~/pkt_ws/mechanistic-steering-vlas`)·HF 캐시 경로를
하드코딩한 **재현 기록**이다. 다른 머신에서 재실행하려면 상단 경로 상수를 수정한다.

## 핵심 결과 (요약; 상세는 docs/steering/16)
- **해석 재현 ✅**: value vector 가 의미 토큰으로 디코드, action-token 이 후반 레이어에 집중
  (L0~18 ~39% → L31 86%, 논문 Fig 2b), 우리 투영이 저자 `up_10` 클러스터와 10/10 일치.
- **스티어링**: fast vs slow 는 유의(paired t-test p=0.029, dz=0.82, 8/10 task)하나 **크기는
  비교방식 의존**(전체속도 +26% / 길이통제 +16% / 논문방식 +38%). fast·slow **둘 다
  baseline 보다 느림**(논문 로봇 관찰 "fast≈baseline" 과 합치). 길이통제 시 효과 ~반감 →
  길이 confound 입증.
- 아티팩트: `artifacts/` (fast/slow 클러스터 yaml, 변위 비교 CSV).
