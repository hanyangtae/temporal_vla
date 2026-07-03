# Dr. VLA — SAE로 VLA feature 해석·steering (reading note)

- 원문: `docs/references/dr_vla.pdf`
- 정식 제목: **"Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models"**
  (Aiden Swann, Lachlain McGranahan, Hugo Buurmeijer, Monroe Kennedy III, Mac Schwager — Stanford, arXiv:2603.19183v1, 2026-03-19)
- 프로젝트 페이지 / 오픈소스 이름: **Dr. VLA** (`drvla.github.io`)
- 우리 관심 관점: GR00T N1.5 latent에서 **outcome(성공/실패) feature를 scene/task/length confound에서 분리**할 수 있는 "Rung 4" 후보 방법으로 평가.

> 주의: 우리 노트에서 "generality vs memorization metric"이라고 부른 것이 이 논문의 핵심이지만,
> 이 metric은 **outcome(성공/실패) metric이 아니다.** "일반화 가능한 primitive feature vs 특정 에피소드
> 암기 feature"를 나누는 metric이다. 이 구분이 우리 문제와 어떻게 겹치고 어긋나는지가 8절의 핵심.

---

## 1. 문제·주장

**문제의식.** VLA는 VLM backbone의 광범위한 semantic 일반화를 로봇 제어로 옮기려 하지만, SFT
(supervised fine-tuning) 이후 언어-following·일반화 능력을 잃고 새 물체/장면/지시에서 실패한다.
LIBERO-PRO[10] 등은 원 프로토콜에서 90%+ 성공률을 내는 모델이 체계적 perturbation에는 거의 0으로
붕괴함을 보여, 이 정책들이 "지각 입력 일반화"가 아니라 **action sequence·환경 layout의 rote
memorization**에 의존한다고 시사했다. 하지만 이 brittleness에 대한 설명은 대부분 행동·일화적
(behavioral/anecdotal)이고 내부 mechanism에 근거하지 않았다.

**주장(기여).**
1. Sparse Autoencoder(SAE)를 VLA **residual stream**에 학습시키면, motion primitive·task-progress·
   language semantic·episode memorization에 대응하는 **해석 가능한 sparse feature basis**가 나온다.
   (샘플링 추정으로 feature의 ~79%가 interpretable — Table 1.)
2. **rollout 없이** fine-tuning 데이터의 activation 통계만으로 각 feature가 "general(전이 가능
   primitive)"인지 "memorized(에피소드 고유)"인지 정량화하는 metric군을 제안.
3. 모든 경우에 **episode-level memorization이 지배적**(feature의 89~99.5%가 memorized로 분류)이지만,
   더 크고 다양한 데이터(DROID) 또는 knowledge insulation(KI)로 fine-tuning하면 general feature 비율이
   증가한다.
4. **개별 SAE feature 방향이 closed-loop 행동에 인과적으로 영향**을 준다(steering 실험, LIBERO).
5. 오픈소스 패키지 **Dr. VLA** (activation 수집 + SAE 학습 + feature 평가 + steering) 공개.

**대상 모델/벤치.**
- Primary: **π0.5** (PaliGemma VLM backbone + 별도 action expert, flow-matching/denoising action
  decoder). 두 fine-tune 변형 π0.5-LIBERO, π0.5-DROID (둘 다 KI로 학습).
- Secondary: **OpenVLA** (Llama2-7B, autoregressive action, LIBERO-Spatial fine-tune) — 아키텍처
  일반성 확인용.
- 벤치: LIBERO(closed-loop steering), DROID(real-world 다양성으로 general feature 발굴).

**메우는 gap.** VLA mechanistic interpretability는 아직 초기이고, 기존 연구는 (i) linear probe로
state decoding(Lu[20], Molinari[21]) 또는 (ii) FFN 방향·contrastive latent로 steering(Häon[22],
Khan[23], Buurmeijer[24])에 그쳤다. Molinari·Buurmeijer는 SAE를 "미래 방향"으로만 언급하고
구현하지 않았고, Khan은 SAE steering을 하지만 feature 해석 가능성을 분석하지 않고 여러 latent
방향에 동시에 작용하는 contrastive 개입을 썼다. 이 논문은 **VLA에 SAE를 실제로 학습시켜, 개별
monosemantic feature가 해석 가능 + general/memorized 구분 가능 + 단일 feature로 인과 steering
가능**함을 처음 보인 것.

---

## 2. SAE 아키텍처·수식 (TopK + AuxK, Gao et al. 2024[12] 기반)

TopK activation[28]로 sparsity를 직접 제어하고, AuxK auxiliary loss로 dead latent 문제를 완화하는
구조. 정확한 수식(원문 Eq 1~3, 12~13):

**(1) per-sample 정규화 (Eq 1).**
입력 activation x ∈ R^d 에 대해:
- 학습된 pre-bias b_pre (초기값 = 학습 activation의 geometric median)를 빼고,
- per-sample scalar mean μ (d_model 차원에 대한 평균)를 빼고,
- ℓ2 정규화:

  x̃ = ((x − b_pre) − μ) / ‖(x − b_pre) − μ‖₂

**(2) encoder (Eq 2).**
  z = ReLU( TopK( W_enc x̃ ) )
- TopK: pre-activation 중 k개 최대값만 남기고 나머지 0.
- post-selection ReLU: 계수 비음수 보장.

**(3) decoder.**
  x̂̃ = W_dec z  (정규화 공간 재구성). 저장해둔 ℓ2 norm·mean·pre-bias를 되돌려 un-normalize → x̂.
- **decoder column은 unit norm 제약** → 각 feature 기여는 오직 activation 계수로 결정.
- encoder/decoder 모두 **bias 항 없음**.

**(4) loss (Eq 3).**
  L = ‖x − x̂‖₂² / C_MSE  +  α · ‖ẽ − ễ_aux‖₂² / C_MSE
- C_MSE = 초기화 시점에 계산한 centered activation의 variance (정규화 상수).
- α = 1/32.
- ẽ = x̃ − x̂̃ : 정규화 공간의 재구성 residual.
- ễ_aux : 이 residual을 top-k_aux **dead latent**만으로 재구성한 것(AuxK). dead 정의 = 최근 500
  optimization step 동안 미발화.

**(5) 초기화·최적화 트릭.**
- encoder 가중치 = decoder 전치의 스케일판: W_enc = W_decᵀ · sqrt(k/n) (n = residual stream 차원).
- decoder gradient는 unit-norm 제약의 tangent plane에 projection(Bricken[16]).
- gradient norm clip = 1.0.

**(6) 폭·sparsity·layer.**
- **expansion ratio(ER) = 1×** (dictionary 크기 = 입력 차원). 이유: 로봇 데이터는 소규모라 큰 ER은
  dead feature만 폭증시키고 interpretability 이득은 비슷 → Fig 7 ablation으로 ER=1 선택. OpenVLA만
  ER=0.5 (dictionary 2048로 π0.5와 맞춤).
- π0.5 PaliGemma layer 0/5/11/17: d=2048, **k=100**.
- π0.5 action expert layer 0/5/11/17: d=1024, **k=64**.
- OpenVLA Llama2 layer 0/8/16/24/31: d=4096.
- 기존 SAE 연구는 보통 단일 중간층에 집중했지만, 이들은 각 subnetwork에서 대략 등간격 4개층(첫/
  이른-중간/늦은-중간/마지막)을 학습. **main 분석은 이른·늦은 중간층(PG5, PG11)에 집중.**

**tap site (정확히, §A.1.3).**
- **residual stream = 전체 transformer block의 출력** (self-attention + MLP 두 sublayer와 각
  residual connection을 모두 거친 뒤). = mechanistic interp의 표준 residual 위치[12].
- PyTorch `register_forward_hook`으로 target decoder layer 모듈에서 캡처.
- **PaliGemma hook은 prefill 1회 pass**(모든 image + instruction 토큰 처리 시) 발화.
- **action expert hook은 action diffusion의 최종 denoising step에서** 발화. (denoise step 축을
  1개만 남긴다 — 우리와의 차이, 8절 참조.)
- **token pooling: hook 내부 모든 토큰을 mean-pool → timestep당 d차원 벡터 1개.** π0.5는 timestep당
  가변 길이 토큰(카메라당 256 × 이미지 = 768 image token + instruct + state token; 실제 실험은
  카메라 2개 사용)을 처리하는데, 이를 전부 pool. 에피소드당 (T, d) 행렬. main text의 모든 SAE
  결과는 이 pooled activation 사용.
  - pooling 이유 2가지: (i) timestep 하나가 곧 obs-action 쌍 = 로봇 행동의 자연 단위이고, 토큰
    시퀀스는 개별 semantic이 약한 image patch 토큰이 지배 → mean-pool이 aggregate state로 압축;
    (ii) storage: per-token은 단일층 2000-episode DROID에 ~3.5TB 필요.
- per-token 변형(§A.7): 저장 문제 완화 위해 카메라별 patch를 mean-pool해 카메라당 1 image 벡터
  (2개) + ~21개 text-token 벡터(각 d=2048). PG5·layer0에서 ER 1×~8×로 별도 SAE 학습.

---

## 3. 학습 셋업

**학습 데이터(어떤 activation·얼마나, Table 3).**
- LIBERO: 1,693 episode / 40 task / **273,465 timestep**, 7-dim action(6-DoF eef + gripper).
  4개 suite(Spatial/Object/Goal/LIBERO-10). Spatial·Object는 10개 task가 base scene 1개를 공유
  → 시각 환경(~20개)이 task 수보다 적고 에피소드 간 유사도 높음(memorization 유리 조건).
- DROID(2k subset): 2,000 episode(1,750 성공 / **250 실패 — 의도적으로 포함**) / 1,545 task /
  **567,088 timestep**, 8-dim action(7 joint pos + gripper). 단, π0.5-DROID 자체는 Pi가 full DROID
  (~75k episode)로 학습; 이 2,000은 **activation 수집·분석 전용**.
- LIBERO-Goal(OpenVLA): 428 episode / 10 task / 52,042 timestep.

**최적화(Table 4).**
- ER=1, active features k=100, k_aux=512, aux loss coeff λ_aux = 1/32,
- lr η = 1e-4, Adam(β1=0.9, β2=0.999), batch 4096, **epoch 100**,
- geometric median 추정 샘플 10,000개(pre-bias 초기화용).

**트릭.**
- AuxK dead-latent 재활용(500-step 미발화 시 dead), decoder unit-norm tangent-plane projection,
  grad clip 1.0, per-sample geometric-median pre-bias.
- **multi-seed robustness(§A.3):** 동일 activation(π0.5 LIBERO PG5)에 seed 7개로 SAE 재학습 →
  주어진 에피소드의 top feature가 seed 간 일관되게 복원됨 → feature가 SAE 상상이 아니라 모델
  표현의 구조임을 확인.
- **embedding baseline(§A.4.2):** π0.5 embedding layer와 **frozen SigLIP**(로봇 데이터 미노출)에도
  SAE 학습 → SigLIP은 event-locked 구조 없이 훨씬 dense → 우리 general feature는 pretrained visual
  표현에서 물려받은 게 아니라 **robotic fine-tuning 중 emergent**.
- 컴퓨트: Stanford Marlowe GPU cluster.

---

## 4. Feature 식별·선택 + "generality vs memorization" metric (우리에게 가장 중요)

수천 개 feature를 수작업으로 다 볼 수 없으니, **per-feature activation 통계 4종**을 정의해 자동
분류한다. 표기: f_j(x_t^(e)) = feature j의 episode e, timestep t 활성값(≥0). T^(e)=에피소드 길이.
E=전체 에피소드. E_j^+ = feature j가 한 번이라도 발화한 에피소드 집합.

**(a) Episode Coverage (Eq 4).**
  c_j = |E_j^+| / |E|
= feature가 최소 1회 발화한 에피소드 비율. 높을수록 다양한 task에 걸쳐 활성 → generality 높음.

**(b) Mean Onset Count (Eq 5–7).**
onset = feature의 inactive→active 전이. 노이즈 억제 위해 threshold τ_on = 0.1로 binary state s
(s0=0) 정의:
  s_t = 1  (f_j(x_t) > τ_on),  0  (f_j(x_t) = 0),  s_{t-1}  (그 외 hysteresis)
per-episode onset count: o_j = Σ_t max(0, s_t − s_{t-1}) (= 0→1 전이 횟수).
active 에피소드에 대해서만 평균(coverage와 분리): ō_j = (1/|E_j^+|) Σ_{e∈E_j^+} o_j^(e).
active 에피소드는 최소 1 onset → ō_j ≥ 1. **general feature는 ō ≫ 1 (bursty, event-driven).**

**(c) Mean Activation Magnitude (Eq 8).**
각 active 에피소드에서 feature의 timestep-최대 활성을 기록, 그 per-episode 최대들의 평균:
  ā_j = (1/|E_j^+|) Σ_{e∈E_j^+} max_t f_j(x_t^(e)).
= 발화 시 전형적 peak 강도.

**(d) Relative Run Length (Eq 9–10).**
run length = onset당 연속 active timestep 평균: r_j = (1/o_j) Σ_t s_t.
에피소드 길이로 정규화: ℓ̄_r,j = (1/|E_j^+|) Σ_{e∈E_j^+} r_j^(e) / T^(e).
**0 근처 = 짧은 transient(general, bursty), 1 근처 = 에피소드 전체 지속(memorized).**
→ **이 축이 사실상 "지속시간/length" 축**이다. 우리 length-confound 문제와 직접 맞닿음(8절).

**general vs memorized 정의(§3.3.1).**
- **General:** 다양한 에피소드에서 semantic하게 일관된 event(잡기·놓기·특정 물체 등장 등)에
  반응. 특징: (i) 높은 ō (bursty, event-locked), (ii) 넓은 coverage c, (iii) 높은 mean active
  fraction. 경험적으로 ō > 1.0, coverage 0.2~0.9(데이터 다양성 의존). **decoder 열이 곧 효과적인
  steering 방향** → residual에 더하면 여러 에피소드에 걸쳐 재현적으로 행동 조절.
- **Memorized:** 특정 에피소드/시각 scene/좁은 task 변형에 튜닝. 특징: (i) 낮은 ō (≈0–1, 지속형),
  (ii) 낮은 coverage(단일 에피소드/동일 scene 소집단), (iii) top 에피소드→다음 사이 max 활성의
  가파른 drop-off. 좁지만 여전히 interpretable.
- 이 둘은 **spectrum**이고 항상 이분되지 않음(예: DROID F322는 특정 사무실 scene의 체스말 잡기를
  주로 암기하지만 다른 scene 체스판에도 약하게 발화).

**수작업 라벨링 = 3단계 시각 검사(§3.3.2).**
- Stage 1 **Activation Viewer**: 한 층의 top 50–100 feature heatmap(행=feature, 열=timestep,
  카메라 프레임과 정렬). bursty(짧고 강한 peak, 행동 event 일치) = general 후보, sustained
  (거의 균일) = memorized 후보.
- Stage 2 **Feature Search index**: 전 데이터셋 SAE activation의 사전 index. 전역 top-activating
  timestep + max 활성 기준 top-10 diverse 에피소드 반환. general이면 여러 에피소드·task에 걸쳐
  고르게, memorized면 한두 에피소드 집중 + 가파른 drop-off.
- Stage 3 **라벨 기준**: general = (1) 활성 burst가 top diverse 에피소드 전반에서 semantic하게
  일관된 event에 정렬 + (2) held-out 에피소드에서도 재현 + (3) 전역 metric이 데이터셋 큰 비율에서
  persistent. memorized = (1) top 에피소드 내 지속 활성 + (2) top 에피소드가 ≤2 scene에 clustering
  + (3) 전역 metric이 작은 subset 발화. 모호하면 제외.

**자동 분류기(§3.3.3, Eq 11).** 4 metric에 대한 logistic regression:
  P(general | m) = σ( β0 + β1·ō + β2·c + β3·ā + β4·ℓ̄_r )
- 단일 reference layer의 **hand-label 30개(general 15 / memorized 15)**로 학습.
- **unnormalized metric 값**에 작동 → 같은 모델 내 층 간 재사용(per-layer 정규화 불필요).
- fine-tuning 데이터셋당 1개(LIBERO, DROID). OpenVLA는 LIBERO 경계 재사용(LIBERO fine-tune이므로).
- LIBERO 계수: β0=−4.20, β1=1.89(ō), β2=1.80(c), β3=0.52(ā), β4=−0.36(ℓ̄_r). 30개에 LOO-CV 100%.
  β1≈β2 → LIBERO는 scene 공유가 많아 bursty + 넓은 coverage 둘 다 필요.
- DROID 계수: β0=−1.78, β1=0.74, β2=2.36, β3=0.35, β4=−1.04. LOO-CV 96.7%. **coverage가 지배
  (β2가 β1의 3.2배)** — 데이터가 다양할수록 넓은 coverage가 generality의 강한 신호. ℓ̄_r 음의
  가중치도 더 큼(−1.04 vs −0.36) — DROID 에피소드가 더 길고 변동 큼(µ=283.5, σ=219.2, CV=0.77
  vs LIBERO µ=161.5, σ=68.2, CV=0.42)이라 run length가 더 강한 판별자.

**핵심: 이 metric은 rollout·성공라벨 없이, fine-tuning 데이터 activation 통계만으로 계산.** →
소규모 표본에서 fabricated/overfit feature를 피하는 그들의 방식 = "이 feature가 여러 에피소드/
scene에 걸쳐 일관되게 같은 event에 발화하는가(coverage·onset)"를 통계적으로 검증하는 것이지,
outcome을 예측하는 것이 아님.

---

## 5. Steering / control

**steering 벡터 (Eq 12–13).**
- feature i의 steering 방향 = decoder 행렬의 i번째 열: v_i = W_dec[:, i] ∈ R^d (unit norm 이미
  보장).
- residual에 스칼라 계수 α로 additive 주입: **y' = y + α · v.**
- **sequence 차원 전체에 broadcast** — 모든 token 위치, **모든 denoising step**에 동일 perturbation.
  (LLM steering과 다른 점: 이 모델은 action을 autoregressive로 뽑지 않음.)
- ⇒ **단일 feature, additive, single-vector 개입.** (우리 conceptor의 multi-dim contrastive 연산자
  h'=h·Mᵀ 와 대비 — 8절.)

**closed-loop 프로토콜(§A.6.1).**
- π0.5: 한 forward pass에서 **50 timestep** action trajectory 예측(**10 denoising step**). steering
  hook은 **10 iter 전부**에 활성(noise→action 정제 전 과정에 개입). 예측 50 중 **첫 5개만 실행 후
  재질의**. steering은 **10번째 timestep부터 시작해 에피소드 내내 지속.**
- OpenVLA: step마다 1 action autoregressive, 매 env step 재질의.

**대표 결과(정성).**
- **F128 (pre-grasp alignment, LIBERO PG5):** steering 시 로봇이 target 물체에 접근하지만 잡지 않고
  **위에서 hover**. 3 task에서 일관 — feature 해석과 정확히 일치.
- **F1902 (transport):** steering 시 관련 물체 잡기를 **완전히 건너뛰고 goal 위치로 직행**(한 task는
  basket과 충돌해 밀어냄). 3 task.
- ⇒ single feature 방향이 semantic 의미와 일관된 인과 행동을 유발함을 입증.

**ablation(§A.6.2, Fig 11) — layer 깊이·종류별 steerability.**
- random SAE decoder 방향 α=300, 층당 10 feature × 5 episode = 50 측정.
- **AE0(action expert 0층)이 가장 steerable**, 깊이 증가로 효과 급감. PaliGemma 층은 같은 깊이에서
  일관되게 약함, **PG L17은 두 모델 모두 변위 0.**
- ⇒ **steering leverage는 이른 action-expert 층에 있고, VLM backbone 늦은 층은 거의 무력.**

**effect size:** 논문은 정량 ΔSR(성공률 변화)을 보고하지 않음 — steering 결과는 궤적/이미지 정성
시연 + EEF 변위(Fig 11)로만 제시. (거대한 caveat, 6·8절.)

---

## 6. 평가·결과·한계

**metric & headline numbers.**
- Interpretability(Table 1, 샘플링 추정): 총 120개 중 95개 interpretable = **79.17%**.
  π0.5 LIBERO PG5 90% / PG11 80% / DROID PG5 85% / PG11 70% / OpenVLA L8 80% / L24 70%.
- Classification(Table 2): **memorized 비율이 압도적.**
  - π0.5 LIBERO PG5: 2044 feature 중 general 32 → **98.43% memorized**. PG avg 97.37%.
  - π0.5 DROID PG5: 2046 중 general 104 → **94.92% memorized**. PG avg 89.19%.
  - OpenVLA LIBERO-Goal L8: 1775 중 general 8 → **99.55% memorized**. LM avg 99.55%.
- **핵심 추세:** 데이터셋 규모·다양성↑ → general 비율↑. LIBERO-GOAL → full LIBERO → DROID로 갈수록
  general share 꾸준히 증가. (절대 memorized 비율은 여전히 높음 — DROID도 LM 데이터 대비 작음.)
- **KI 효과(§4.5, Fig 6):** fine-tuning step↑ 시 coverage↓·run length↑(generality 감소)인데 **KI는
  이 추세를 역전**. DROID PG5 median P(general): 0.190(10k) / 0.181(30k) / 0.179(60k) / 0.181(90k)
  / **0.206(KI)**. 미묘하지만 feature metric이 rollout 없는 **train-time 일반화 proxy** 가능성 시사
  ("weakly suggest", 더 엄밀 평가 필요라고 스스로 유보).

**failure mode / confound(저자가 인정).**
- **§6 Limitations 핵심:** "meaningful top activation ≠ reliable steerability." 많은 clean feature가
  steering 시 제한적/예측 불가한 인과 효과. 가설: (i) flow-matching의 nonlinear downstream 상호작용,
  (ii) predictive ≠ causal.
- 일부 feature는 general/memorized 이분에 안 맞음. under-classification 예(§A.5.1):
  - **F1939(LIBERO PG5):** ō=1.00, c=0.732, ā=0.037, ℓ̄_r=0.127. coverage 73%인데 onset이 정확히
    1.0이라 memorized로 오분류. 실제로는 모든 에피소드 앞 20 timestep에 발화 = 로봇 **"home"
    자세**(scene·task 불변) → 눈으로는 general. **한 에피소드당 1회 발화 패턴이 general을 못 잡음.**
  - **F1381(DROID PG5):** ō=1.00, c=0.226, ℓ̄_r=0.990. lid 종류·scene 무관하게 뚜껑 잡기에 발화
    ("lid" 포함 task 135개 중 116개 발화, 86% recall)인데, lid 에피소드가 데이터의 6.7%뿐이라
    coverage가 경계 아래 + 지속형(ℓ̄_r≈0.99)이라 memorized로 오분류. → **작은 subset에 걸친
    coherent feature를 memorized로 오판.** 해결엔 diversity-aware coverage 정규화 또는 cross-scene
    consistency metric이 필요하다고 인정.
- 기타: VLM 연구 대비 데이터가 orders of magnitude 적음. main은 **mean-pooled token**(per-token은
  덜 interpretable, §A.7). **하드웨어 평가 없음**(sim only). 고성공률 general 정책 부재도
  steerability 저해 가능.

---

## 7. 오픈소스 toolkit

- **Dr. VLA** 패키지 (project page: drvla.github.io) — **activation 수집 + SAE 학습 + feature 평가 +
  policy steering**을 위한 오픈소스 + user-friendly 인터페이스.
- 웹 상 도구: **Interactive Feature Browser**("labeled" 태그로 LIBERO 10개 feature 상세), **Activation
  Viewer**(에피소드 heatmap), **Feature Search**(전역 top-activating timestep/에피소드 index),
  Fig 3/4의 인터랙티브 버전.
- 직접 재사용 가능 후보(구조상): TopK+AuxK SAE 학습 루프, per-feature 4-metric 계산, logistic
  generality 분류기, decoder-column additive steering hook, activation 수집 forward-hook.
- **라이선스: PDF 본문에 명시 없음** — 실제 라이선스/코드 구조/API는 repo(drvla.github.io 링크)를
  직접 확인해야 함. (본 노트는 PDF만으로 작성했으므로 이 부분은 미확인이라고 명시.)

---

## 8. 우리 적용성 (GR00T N1.5, outcome-vs-confound 분리 관점)

우리 세팅 요약: GR00T N1.5 = **Eagle-LM(VLM) → VL self-attention → DiT action head** (직렬).
latent = DiT block residual **[L=7 layer, K=4 denoise step, D=1536]** + VL **[2048]**. event-anchored
phase(reach→transport→insert-settle). 메인 method = contrastive **conceptor** (C_steer =
C_success ∧ ¬C_failure, h'=h·Mᵀ). 방금 확립한 문제: dense DiT/VL latent의 성공/실패 분리가
**confound** — scene/task를 AUROC=1.0, length/phase를 강하게 인코딩하고, scene+length 통제 후
genuine outcome은 permutation-null(~0.9) 수준(scene당 실패 3–4개라 검정력 낮음).

### 8.1 tap site 대응 (그들의 L5의 우리 analog)

| 그들(π0.5) | 우리(GR00T N1.5) | 비고 |
|---|---|---|
| PaliGemma(VLM backbone) 전체 block residual, **PG5**(이른-중간)·PG11 = 가장 interpretable/general | **Eagle-LM residual, 이른-중간 층** | 우리 VL[2048]은 차원(d=2048)이 PaliGemma와 우연히 일치하지만 backbone LM이 다름. VL 캡처가 **어느 Eagle 층/pool인지 확정 필요**. "goal/what" pathway. |
| action expert(AE) block residual, **최종 denoise step만, mean-pool** | **DiT block residual [L=7, K=4, D=1536]** | "motor/how" pathway. 그들은 K축을 최종 1 step으로 접었지만 우리는 K=4 전부 보유 → **denoise 축 phase 자원 추가.** |
| AE0가 가장 steerable, PG L17=0 | 이른 DiT 층이 steering leverage 가능성 | Fig 11 추세를 GR00T에 옮기면 **이른 DiT 층부터** steering 탐색이 합리적. |

⇒ **"L5의 analog"은 backbone(VL) 관점에선 Eagle-LM 이른-중간 층**이고, **motor(DiT) 관점에선 DiT
블록 residual**이다. 그들의 general "motion-primitive" feature는 AE 층에서, 해석 좋은 semantic
feature는 PG5/PG11에서 나왔다는 점에 유의 — 우리도 **VL과 DiT를 따로 SAE 학습**하는 게 자연스럽고,
이는 우리 pathway-분리 전략과 정합.

**hook/수집 인프라는 부분적으로 이미 존재:** 우리 메모리(groot-robocasa-serve-path)상 N1.5 lerobot
HTTP 경로에 capture·steer 배선이 있고 hook 지점은 action_head DiT block. Dr.VLA는 표준 forward-hook을
쓰므로 우리 hook 인프라와 개념 호환. denoise 축·chunk(predict16/execute5)에 맞춘 timestep 정렬만
추가로 정의하면 됨.

### 8.2 generality metric이 우리 "scene 3–4 실패" 검정력 문제를 도와주나? (솔직한 판정)

**직접적으로는 아니다.** 그들의 metric은 outcome(성공/실패) metric이 아니라 general vs
memorized(scene/episode 암기) metric이다. 그들 결과에는 "이 궤적은 실패할 것" feature가 없다.
DROID 실패 250개를 넣고도 성공/실패 분리는 분석하지 않았다. 따라서 **outcome isolation을 위한
직접 방법을 이 논문은 제공하지 않는다.**

**그러나 간접적으로 강하게 유용하다 — 문제를 다른 공간으로 재구성해준다:**
1. **confound가 곧 그들의 memorized feature다.** 우리 dense latent가 scene/task를 AUROC=1.0으로
   인코딩한다는 것은, 그 latent가 memorized(scene/episode) 방향으로 지배됨을 뜻한다. SAE는 이
   dense latent를 (a) memorized scene/task feature와 (b) general motion-primitive/task-progress
   feature로 **분해(disentangle)**한다. 즉 **scene confound가 별도 feature로 factor-out**된다. →
   dense space에서 scene와 싸우는 대신, scene feature를 빼고 outcome을 general/progress feature
   공간에서 볼 수 있다. 이게 "Rung 4"로서의 진짜 가치.
2. **length confound에 원리적 축을 제공한다.** relative run length ℓ̄_r 는 사실상 지속시간 축이다.
   우리 실패=항상 timeout(길다)/성공=조기종료(짧다)라 length가 라벨을 결정하는 문제(seen18
   length-confound)와 정확히 맞닿는다. SAE feature 중 ℓ̄_r 이 높은(sustained) 것은 length-상관
   feature로 태깅되므로, **outcome 판정 시 이 feature들을 배제/통제**하는 원리적 필터가 된다.
3. **phase 축이 native로 지원된다.** SAE는 per-timestep(시간 pool 안 함)이고 onset/run-length가
   temporal metric이다. 우리 phase-matched(event-anchored) 접근과 궁합이 좋다 — F1129(grasp/place),
   F1902(transport), F445(task completion) 같은 **phase-progress feature가 바로 우리가 원하는
   online phase 신호 공급원** 후보다.

**다만 검정력 문제 자체(scene당 3–4 실패)는 SAE가 마법처럼 풀지 않는다.** SAE는 unsupervised
disentanglement일 뿐, "실패 방향"을 자동으로 주지 않는다. 우리가 추가로 해야 할 일:
**SAE feature activation을 outcome label에 회귀/대조**(그들의 generality 회귀를 outcome 회귀로 치환).
sparse·monosemantic feature에서 scene은 이미 별도 feature로 빠졌으므로, outcome 회귀의 검정력은
dense space보다 (원리적으로) 높아질 수 있다 — 다만 이는 가설이고, 실측 검증 필요.

### 8.3 method 합성 아이디어 (conceptor × SAE)

- 그들의 steering은 single-feature additive(y'=y+αv). 우리 method는 contrastive conceptor(multi-dim).
  **둘을 합치면:** raw residual 대신 **SAE feature 공간에서 conceptor를 fit** → C_success ∧ ¬C_failure를
  disentangled·scene-free 좌표에서 계산 → scene superposition 오염 감소. 이게 우리 confound
  분리 목표에 가장 직접적인 synthesis.
- 반대 방향 리스크: general feature가 전체의 1~8%뿐이고, "predictive ≠ steerable"(그들이 명시). 즉
  **outcome-상관 feature를 찾아도 그것으로 steer가 듣는다는 보장은 없다** — 우리의 인과 재측정
  (ΔSR)이 여전히 최종 판정이어야 함.

### 8.4 필요한 인프라 (build list)

- (있음/부분) GR00T VL·DiT residual forward-hook 수집(lerobot 경로 배선 존재). denoise K·chunk 정렬
  정의 추가.
- (구현) TopK+AuxK SAE 학습(표준, 중간 규모 코드; Dr.VLA 코드 이식 검토 — 라이선스 확인 후).
- (구현) per-feature 4-metric + logistic 분류기(사소).
- (우리 확장, 핵심) **outcome 회귀/대조 레이어**: feature activation vs 성공라벨, scene·length를
  covariate로 통제 (그들이 안 한 부분 = 우리 niche).
- (있음) decoder-column / conceptor steering hook.
- 데이터: SAE 학습엔 대량 activation 필요(그들은 27만~57만 timestep). 우리 raw_rollouts로 충당
  가능하나, 원격 노드 데이터 위치·CPU/저장 budget 고려.

### 8.5 pros / cons 요약 (outcome-vs-confound 분리 목표 기준)

**Pros**
- dense latent를 monosemantic feature로 분해 → **scene/task memorization을 별도 feature로 factor-out**
  (우리 AUROC=1.0 scene confound에 직접 대응).
- **rollout 없이** 계산되는 generality/coverage/run-length metric → train-time·저비용.
- **run length = 원리적 length 축**, onset/temporal = 원리적 phase 축 → 우리 두 confound(length·phase)에
  이름 붙은 통제 변수 제공.
- per-timestep·event-locked → 우리 event-anchored phase, chunk(predict16/execute5)와 정합.
- conceptor를 feature 공간으로 옮길 자연스러운 synthesis 경로.

**Cons / 미해결**
- **outcome(성공/실패) metric을 직접 제공하지 않음** — 우리가 회귀 레이어를 추가해야 함.
- general feature가 극소수(1~8%), 나머지는 memorized → 깨끗한 outcome feature 확보가 보장되지 않음.
- 저자 스스로 **predictive ≠ steerable**, flow-matching downstream nonlinearity 경고 → steer가 듣는지는
  별도 인과 검증 필요(우리 ΔSR).
- **정량 ΔSR 결과 없음**(정성 시연 + EEF 변위만) → steering 효과 크기 근거 약함.
- generality 분류기가 single-onset general feature(예: home-position, 소수 subset lid feature)를
  오분류 → 우리 outcome feature도 유사 오분류 위험, diversity/length-aware 정규화 필요.
- Eagle-LM은 PaliGemma가 아님 → 그들 layer 선택(PG5)이 그대로 안 옮겨짐. GR00T에서 층 sweep 필요.
- SAE 학습에 대량 activation·저장(그들 per-token은 단일층 3.5TB) → mean-pool 채택 필수, per-token은
  덜 interpretable.

**한 줄 판정:** Dr.VLA는 우리 confound 문제를 *푸는* 방법이 아니라, 문제를 *scene/length가 별도
feature로 분리된 disentangled 공간으로 옮겨주는* 도구다. outcome 분리 자체는 우리가 그 위에
supervised 레이어(feature↔outcome 회귀 + scene/length 통제)를 얹어야 완성되며, 그 부분이 정확히
우리의 미점유 niche(내부 latent × online × 실패 TYPE × phase-matched)와 겹친다. 검정력 향상은
"원리적으로 기대되지만 실측 미검증"으로 남는다.
