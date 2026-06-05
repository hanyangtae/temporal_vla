# Latent steering 방법론 survey (GR00T N1.6 적용 후보)

작성 맥락: pathway-resolved failure steering plan(Phase 0). 헤드라인은 **conceptor**
(`C_steer = C_success ∧ ¬C_failure`, COAST) 유지하되, 나중에 비교·교체할 후보를 정리한다.
판단 기준 4가지:

1. **적용 형태** — additive(`h' = h + αv`) vs projective/multiplicative(`h' = h·Mᵀ`).
2. **GR00T fragility 적합성** — NOTALL: GR00T DiT feature는 fragile(−68pp@9× amp),
   boosting이 bidirectionally destructive(G.3), motor program은 trajectory 초기에 commit.
   → **projective·suppressive·early·소강도**가 안전. additive boosting은 위험.
3. **구현 난이도** — 우리 hook infra(`scripts/serve/steering_hooks.py`) 재사용 가능성.
4. **pathway 적합** — goal(VL/vlln) vs motor(DiT). VL은 token-pool, DiT는 per-token.

우리 hook 인프라는 이미 `pathway∈{dit, vl}` 와 `token_select∈{last_horizon, all}` 를
지원한다(`ConceptorSteering`). 아래 방법 대부분은 이 hook 의 적용식만 바꾸면 된다.

---

## A. 헤드라인 — Conceptor steering (현 방식, COAST)

- **형태**: projective/multiplicative. `M = (1-β)I + β·C_steer`, `h' = h·Mᵀ`.
  C_steer = C_success ∧ ¬C_failure (success 부분공간 유지 + failure 부분공간 억제).
- **fragility**: ★★★ 좋음. multiplicative gate라 magnitude 보존적, β로 강도 연속 조절,
  suppressive 성격. 우리 prior DiT 실험에서 SR 하락했지만 그건 pathway/표상 문제로 해석.
- **구현**: 이미 구현 (`src/conceptor/`, `ConceptorSteering`, `fit_conceptor_steering.py`).
- **pathway**: dit/vl 모두 적용 가능. plan Phase 4 의 기본.
- 참고: Postmus & Abreu 2024 (arXiv 2410.16314), COAST (arXiv 2605.17144).

## B. CAA — Contrastive Activation Addition

- **형태**: additive. `v = mean(h_success) − mean(h_failure)`, `h' = h + α·v`. 단일 방향.
- **fragility**: ★☆☆ 주의. additive boosting → GR00T에서 위험(NOTALL G.3). 단 **suppressive
  방향**(`h' = h − α·v_failure`, failure 평균에서 멀어지기)으로 쓰면 위험↓. 소 α 필수.
- **구현**: ★★★ 매우 쉬움. hook 에서 M 대신 (mean diff 벡터, α) 적용. conceptor의 1-D 특수화.
- **pathway**: vl(goal)에서 "성공 goal 방향으로" 미는 데 직관적. baseline 비교군으로 적합.
- 참고: Rimsky 2024 (ACL, arXiv 2406.00154), Turner 2023 activation addition (2308.10248).

## C. SAE-guided / feature-guided activation additions

- **형태**: additive in SAE latent. per-token SAE 학습 후 concept feature를 ablate/boost.
  `h_mod = h + (W_e·z_modified − W_e·z)` (NOTALL D.3).
- **fragility**: ★☆☆ NOTALL 직접 보고 — ablation은 관대(p=0.975), boosting은 7×+에서 파괴.
  GR00T DiT는 per-token 필수, VL-SA는 pooling 이득.
- **구현**: ★☆☆ 무거움 — 모델당 per-token SAE 수십 개 학습(NOTALL: GR00T 68개). frequency-
  weighted contrastive selection(score = Cohen's d × freq)으로 manipulation concept 추출.
- **pathway**: concept-level 해석 가능성이 강점(자동 실패데이터 수집 Phase 5에 유용). steering
  자체는 약할 수 있음. 별도 큰 투자.
- 참고: Hatton 2025 (CoRL, VLA SAE steering), Khan 2025 (sparse latent dir), Soo 2025,
  Gao 2024 (TopK SAE).

## D. Subspace / cross-task activation injection (NOTALL E.1)

- **형태**: 직접 치환. success rollout의 goal-subspace(예: LDA top-k dim, NOTALL은 20/1024)
  를 실패 episode forward에 주입. conceptor 불필요.
- **fragility**: ★★☆ NOTALL E.1: full injection은 task 파괴하지만 **goal-subspace(2% dim)만
  주입하면 task 보존**(100% on 3/5 pairs). 좁은 subspace 주입이 안전.
- **구현**: ★★☆ 중간 — source activation 저장 + 주입 hook + subspace 식별(LDA/probe). 우리
  hook 으로 `h'[:, sel_dims] = h_src[:, sel_dims]` 형태.
- **pathway**: vl(goal) subspace 주입이 NOTALL "spatially-bound motor program" 가설 검증과
  직결. **인과성 직접 측정**(우회 없음)이 큰 장점. Phase 4 강력한 보조 실험.
- 참고: NOTALL §4.3, E.1; Meng 2022 (activation patching).

## E. SteerVLM / lightweight learned steering

- **형태**: 경량 학습된 steering module(activation 조정). additive지만 학습된 gate.
- **fragility**: ★★☆ 학습 기반이라 강도 조절 가능. 단 추가 학습 필요(우리 "백본 추가학습 없음"
  원칙과 긴장 — steering module만 학습이면 허용 범위 검토).
- **구현**: ★☆☆ 별도 학습 루프 필요.
- **pathway**: 범용. 우선순위 낮음.
- 참고: Sivakumar 2025 (SteerVLM, arXiv 2510.26769).

---

## 권장 적용 순서

1. **Conceptor (A)** — plan Phase 4 헤드라인. dit/vl pathway × β-sweep.
2. **CAA suppressive (B)** — 같은 hook로 즉시 가능한 baseline 비교군. conceptor가 단일벡터
   대비 우월한지 검증(우리 메모리: multi-dim contrastive가 맞다 — 이를 재확인).
3. **Subspace injection (D)** — 인과성 직접 검증 + spatially-bound 가설. Phase 4 보조.
4. **SAE (C)** — Phase 5(자동 실패데이터/concept 해석)와 묶어 별도 투자.
5. **SteerVLM (E)** — 후순위.

공통 원칙(NOTALL 근거): **projective·suppressive·early·소강도**, additive boosting 회피,
VL=token-pool / DiT=per-token, β=0 sanity, Wilson CI로 ΔSR 검증.
