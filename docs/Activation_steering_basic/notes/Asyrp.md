# Diffusion Models Already Have a Semantic Latent Space (Asyrp) (Kwon et al. 2023)

- 출처: arXiv:2210.10960 (v2, 2023-03-29) · Yonsei University AI학과 (Mingi Kwon, Jaeseok Jeong, Youngjung Uh) · ICLR 2023
- PDF: `docs/Activation_steering_basic/Asyrp_2210.10960.pdf`
- §5파트: 이미지 생성 도메인의 진짜 activation-space steering(연구단계, 도구탑재 미확인)
- 3축: 쓰기(방향 f_t는 CLIP loss로 사전 최적화·학습, per-attribute) · 적용은 inference(매 denoising step에 Δh_t를 더함) · backbone 완전 frozen(사전학습 diffusion U-Net 그대로, 작은 보조 net만 학습)
- 한줄 역할: LLM residual stream steering과 개입 대상이 가장 유사한 이미지판 사례 — noise-estimate(ε) 공간이 아니라 U-Net bottleneck **hidden state h 자체**를 직접 수정해 semantic 편집이 가능함을 최초로 보임.

## 문제·동기

Diffusion model은 x_T→x_0 결정적 역과정(DDIM)으로 거의 완벽한 실제 이미지 inversion이 가능해 이미지 편집에 유리하지만, "semantic latent space"가 없어 세밀한 제어가 어렵다. 기존 접근: (1) image guidance(모호함, 크기 제어 불가), (2) classifier guidance(별도 classifier 학습 + 매 step gradient 계산 비용), (3) DiffusionCLIP류 전체 모델 파인튜닝(속성별 모델 여러 개 필요). GAN latent space처럼 "방향 하나 = 속성 하나"인 편집 가능한 공간이 diffusion에도 있는지가 질문. 저자들은 노이즈 예측치 ε_t를 직접 shift하는 가장 단순한 방법이 왜 실패하는지부터 이론적으로 규명한다(Theorem 1): reverse step x_{t-1} = √α_{t-1}·P_t(ε_t) + D_t(ε_t)에서 P_t("predicted x0")와 D_t("direction pointing to x_t") 양쪽에 동일한 shift Δε_t가 들어가면 서로 상쇄(destructive interference)되어 x_0가 거의 변하지 않는다.

## 핵심 아이디어(h-space, asymmetric reverse process)

상쇄를 깨려면 대칭을 깨면 된다: P_t에는 shift된 ε~_t를 넣고 D_t에는 원본 ε_t를 그대로 유지 — x_{t-1} = √α_{t-1}·P_t(ε~_t) + D_t(ε_t). 이렇게 하면 반대편(D_t)이 원래 궤적을 유지하므로 P_t 쪽 변화가 상쇄되지 않고 x_0에 반영된다. 이 asymmetric reverse process(Asyrp)를 어디에 적용할지가 다음 질문 — ε_t space 자체에 최적화된 Δε를 찾아도 편집은 되지만 "속성 하나에 방향 하나"의 좋은 성질(아래)이 없다. 대신 ε_θ를 구현하는 U-Net의 **bottleneck(가장 깊은 layer, 8번째, skip-connection 영향 없음, 최소 spatial resolution·최고수준 semantics)** feature map h_t를 h-space로 채택하고, 여기에 Δh_t를 더한다: ε_θ(x_t|Δh_t). h-space는 ε-space가 갖지 못하는 5가지 성질을 보인다 — homogeneity(동일 Δh가 다른 샘플에도 같은 효과), linearity(스케일링이 곧 편집 강도, 미학습 음의 스케일까지 일반화), robustness(랜덤 노이즈를 더해도 realistic 유지), timestep 간 consistency(Δh_t가 t에 걸쳐 거의 불변 → Δh^mean/Δh^global로 대체 가능).

## 방법(Δh를 경량넷으로 학습, backbone frozen)

- 매 t마다 Δh_t를 직접 최적화하는 것은 학습률 스케줄링이 예민하고 느림 → 대신 implicit function f_t(h_t)를 학습: 1x1 conv 2개 + timestep sinusoidal embedding으로 구성된 경량 네트워크(파라미터 매우 작음, group norm+swish, DDPM 스타일).
- Loss: directional CLIP loss(Gal et al. StyleGAN-NADA 방식, Δ이미지·Δ텍스트 cosine 정렬) + L1 재구성 정규화(편집된 predicted-x0 vs 원본 predicted-x0). Backbone U-Net은 완전 frozen, f_t만 gradient 업데이트.
- Attribute 1개당 f_t 1개, 1000개 샘플·1 epoch, accelerated DDIM subsequence(S=40)로 RTX3090 3장 약 20분 학습 — 이후 임의 길이 S~로 일반화(정규화 Δh~_τ~ = f_τ~(h_τ~)·S/S~).
- 전체 생성 과정은 3단계로 설계: (1) editing interval [T, t_edit] — editing strength ξ_t = LPIPS(x,P_T) − LPIPS(x,P_t)를 재서 LPIPS(x,P_tedit)=0.33이 되는 가장 짧은 편집 구간 산출(고차원 구도가 이미 결정된 지점), (2) traditional DDIM denoising, (3) quality boosting [t_boost, 0] — quality deficiency γ_t = LPIPS(x,x_t)=1.2 지점부터 stochastic noise(η=1) 주입해 DDIM 특유의 detail 손실을 최소 content 변화로 복구.

## 실험·결과

- CelebA-HQ/LSUN-church/LSUN-bedroom(DDPM++), AFHQ-dog(iDDPM), METFACES(ADM+P2) — 아키텍처·데이터셋 무관 재현. Church→department store/factory/temple, 웃지 않는 견종(Poodle/Yorkshire)의 smiling 등 학습 데이터에 없는 unseen-domain 속성도 backbone 재학습 없이 합성.
- 80명 user study(vs DiffusionCLIP 전체모델 파인튜닝): CelebA-HQ in-domain overall 94.92% vs 5.08%, unseen-domain 63.13% vs 36.88%, LSUN-church 76.81% vs 23.19% — 전 항목 우세.
- 정량 지표 Sdir(directional CLIP similarity)/SC(segmentation consistency)도 대부분 Asyrp 우세(SC는 DiffusionCLIP이 종종 높으나 이는 구조를 거의 안 바꾸는 texture-only 편집의 부작용으로 해석).
- Ablation: U-Net 16개 layer 전수 조사 결과 8번째 layer(bottleneck)가 유일하게 손상 없는 의미 편집을 지원(1~6층은 변화 거의 없음, 9층 이후는 심한 artifact).

## §5(산업)에서의 위치(연구단계, 도구탑재 미확인)

- 논문·공식 코드(GitHub `kwonminki/Asyrp_official`, 프로젝트 페이지)만 공개 — SEGA(Diffusers `SemanticStableDiffusionPipeline`)처럼 표준 라이브러리·상용 도구에 정식 편입됐다는 근거는 논문에 없다. 순수 연구 프로토타입 단계.
- 다만 h-space 발견 자체가 이후 이미지 편집 생태계(Prompt-to-Prompt류 외의 "bottleneck feature 편집" 계열, DiffusionCLIP 후속 비교대상)에 자주 인용되는 참조점이 됐다는 점에서 간접적 영향력은 크다 — "도구화되지 않았지만 방법론적 표준을 만든" 사례.
- 저자 스스로 Conclusion에서 "guidance/finetuning 기법과 결합"을 향후 연구로 제시 — 실제 프로덕션 배치보다 향후 연구 확장이 논문의 관심사.

## 우리 프로젝트 연결(hidden-state 개입 대응)

- Asyrp의 h-space(U-Net bottleneck feature map)는 우리가 VLA에서 개입하는 DiT/VL-SA hidden state와 개입 **대상**이 동형이다 — 둘 다 "출력 직전의 raw noise/logit 공간"이 아니라 "네트워크 내부, 압축된 semantic 표현"에서 직접 벡터 연산을 가한다. SEGA(ε-space, CFG 항)나 텍스트 prompt engineering과 달리 진짜 hidden-state steering이라는 점에서 우리 방법과 §5 분류축이 가장 가깝다.
- 다만 연산 성격이 다르다: Asyrp의 Δh_t는 CLIP directional loss로 **gradient 최적화해 학습된 방향**(속성 1개당 소형 네트워크 훈련 필요)이고, 우리 conceptor(C_steer = C_success ∧ ¬C_failure)는 succ/fail 활성화 분포의 공분산에서 **closed-form으로 fit한 contrastive subspace 연산자**다 — 학습 여부(gradient descent vs 통계적 fit) 축에서 갈린다. ActAdd/CAA(raw mean-difference, 학습 없음)와도 다른 중간 지점: Asyrp는 "학습되지만 backbone은 안 건드리는" 세 번째 범주.
- Asyrp의 "editing interval [T, t_edit]만 개입"은 우리 phase-matched steering 문제의식과 표면적으로 닮았지만, 그 t는 **단일 이미지 생성 내부의 denoising step**(COAST/우리 taxonomy의 denoising-step 축)이지 우리가 다루는 **rollout task-phase 축**(다중 action-step에 걸친 실제 시간)이 아니다. Asyrp는 한 번의 forward 생성이 끝나면 개입도 끝나므로 "online 실패 인식" 개념 자체가 없다.
- h-space의 timestep 간 consistency(Δh_t ≈ Δh^mean ≈ Δh^global)는 우리가 원하는 "phase-invariant steering 방향이 존재하는가"라는 질문과 유사한 형태이나, Asyrp에서는 소수(≤1000) denoising step 안의 현상이고 우리는 rollout 길이·실패 유형에 따라 분포 자체가 바뀌는 훨씬 어려운 setting이다.

## 면접 포인트(Q→A)

Q1. Asyrp가 activation steering 계보에서 왜 중요한가?
A. 이미지 도메인에서 "noise-estimate 공간이 아니라 네트워크 내부 hidden state(U-Net bottleneck)에 직접 개입해야 의미 편집이 된다"를 이론(Theorem 1, destructive interference)과 실험으로 처음 규명했다. 이는 LLM의 residual stream steering이 "출력 로짓이 아니라 중간 hidden state"를 건드리는 것과 정확히 같은 통찰의 이미지판이다.
Q2. 왜 ε-space가 아니라 h-space, 왜 symmetric이 아니라 asymmetric인가?
A. ε_t를 그대로(대칭적으로) shift하면 P_t와 D_t 양쪽에 반영돼 서로 상쇄된다(Theorem 1 증명). Asyrp는 P_t만 shift하고 D_t는 원본 유지해 상쇄를 깬다. 그 shift 지점을 ε 전체가 아니라 U-Net bottleneck인 h로 좁히면 homogeneity/linearity/robustness/consistency라는 "속성=방향"에 필요한 성질이 생긴다(다른 layer에서는 이 성질이 없거나 약함, App D.3 ablation).
Q3. Δh는 어떻게 얻나, 왜 매 timestep 직접 최적화하지 않는가?
A. 속성마다, timestep마다 Δh_t를 직접 gradient 최적화하면 학습률 스케줄링이 예민하고 느리다. 대신 timestep을 입력으로 받는 소형 implicit function f_t(2-layer 1x1 conv)를 한 번 학습해 임의 t·h에 대해 Δh_t를 생성하게 하면, 학습이 안정적이고 다른 subsequence 길이에도 일반화된다.
Q4. 우리 conceptor 방법과 뭐가 다른가?
A. (1) 방향을 얻는 방식: Asyrp는 CLIP loss로 gradient 최적화해 학습(속성당 소형 네트워크 필요)하는 반면 우리는 succ/fail 활성화 공분산에서 closed-form fit. (2) 시간축: Asyrp의 "phase"는 한 이미지 생성 내부의 denoising step이고 우리는 rollout task-phase(다중 action-step). Asyrp는 forward pass 1회로 끝나 online 실패 인식이라는 문제 자체가 없다.

## 한계·비판

- attribute마다 λ_CLIP, λ_recon, t_edit, t_boost 등 하이퍼파라미터를 손튜닝해야 하고(App. J.1 표), LPIPS 임계값(0.33, 1.2)도 경험적으로 고정한 magic number — "backbone frozen"이 "튜닝 없음"을 뜻하지 않는다.
- 새 속성마다 f_t를 새로 학습해야 한다(SEGA처럼 텍스트만 바꿔 즉석 적용 불가) — 도구 생태계로 확장하려면 오프라인 학습 스텝이 항상 선행돼야 하는 scalability 제약.
- 저자 스스로 인정: 전체 스타일이나 주변 객체(peripheral object) 변경은 잘 안 되고 주 객체 속성 편집에 국한된다.
- 평가가 CLIP 기반 proxy(Sdir)·segmentation proxy(SC)·인간 user study에 크게 의존 — 정량 근거가 여전히 간접적.
- 단일 forward diffusion 생성이라 "실패 후 온라인 복구"·"진행 상태 추적" 개념이 없다 — rollout 중 실패를 감지해 즉시 steering해야 하는 우리 문제로 옮기려면 시간축 자체를 새로 설계해야 한다.
- Ethics: 저자도 동일 메커니즘이 disinformation/propaganda 생성에 악용될 수 있음을 인정.
