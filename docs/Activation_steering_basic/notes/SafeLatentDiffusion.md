# Safe Latent Diffusion (Schramowski et al. 2023)

- 출처: arXiv:2211.05105 (v4, 2023-04-26) · DFKI/Hessian.AI/TU Darmstadt/Aleph Alpha/LAION (Patrick
  Schramowski, Manuel Brack, Björn Deiseroth, Kristian Kersting) · CVPR 2023
- PDF: `docs/Activation_steering_basic/SafeLatentDiffusion_2211.05105.pdf`
- §5파트: 이미지 생성 도메인 "안전 gate로서의 activation steering" — SEGA 수학의 안전 특수화(먼저
  나온 형 논문)
- 3축: 쓰기(write, additive score 항이나 부호가 항상 "부적절 개념으로부터 멀어짐"에 고정) ·
  inference-time(파인튜닝 없음, 매 denoising step 즉석 계산) · 도구탑재(Diffusers 공식 파이프라인
  `StableDiffusionPipelineSafe`, 단 표준 파이프라인과 별개의 opt-in 클래스)
- 한줄 역할: SD 기본 내장 post-hoc CLIP 기반 NSFW safety_checker(생성 후 이미지를 사후 검열·흔히
  비활성화됨)와 달리, 생성 도중 latent space에서 부적절 개념 방향으로부터 score를 실시간으로 밀어내는
  inference-time steering — "안전 gate로서의 활성화 steering"의 이미지 원조 사례.

## 문제·동기

- SD는 LAION-5B(필터링 안 된 웹 스크랩)로 학습돼 폭력·노출·혐오 등 부적절 콘텐츠를 재생산하며, 심지어
  명시적으로 요청하지 않은 무해해 보이는 프롬프트에서도 생성된다(Fig.1: "묵시록의 네 기사(여성)"류
  프롬프트가 60% 확률로 부적절 이미지 생성).
- ethnic bias 실증: `<country> body` 프롬프트로 50개국 비교 시 일본이 75%+ 노출 확률(글로벌 평균
  35%) — LAION의 reporting bias가 학습을 거쳐 SD 생성물에 그대로 전이됨을 보임.
- 기존 대응의 한계: (1) 데이터 필터링(학습 전 조치, 되돌릴 수 없고 능력 손실 우려), (2) post-hoc NSFW
  safety_checker(생성 후 이미지 분류·차단, SD 기본 내장이나 코드 한 줄로 비활성화 가능) — 저자들이
  실제로 lexica.art 배포판에서 이 checker가 꺼진 채 서비스되는 사례를 발견, "끄기 쉬운 사후 필터"의
  구조적 한계를 지적.
- 목표: 추가 파인튜닝 없이, 모델이 사전학습으로 이미 획득한 "부적절함"에 대한 지식 자체를 이용해 생성
  과정 중에 개입한다.

## 핵심 아이디어

- Classifier-free guidance(CFG)가 이미 "무조건 추정치를 조건부 추정치 방향으로 미는" 연산이므로,
  여기에 세 번째 항을 추가한다: 부적절 개념 S로 조건화한 추정치 방향으로부터는 반대로 민다.
- 즉 "prompt 방향으로 당김 + unsafe-concept 방향으로부터는 밀어냄"을 한 식에 통합 — 뒤에 나온 SEGA의
  범용 concept-arithmetic(임의 텍스트 방향, 부호 자유)을 "안전 개념 하나, 방향은 항상 음(-)"으로
  좁힌 특수 케이스에 해당한다.
- SLD와 SEGA는 동일 저자군(Schramowski/Brack/Kersting)이 같은 CFG 확장 수학을 썼다: SLD(2211.05105,
  2022-11, 본 논문)가 먼저 안전 응용으로 냈고, SEGA(2301.12247, 2023-01)가 범용 semantic guidance로
  일반화했다 — 형(SLD)-아우(SEGA) 관계.

## 방법(안전 개념 방향으로 score 밀어내기, SEGA와 자매)

- CFG 기본식: ε~θ(zt,cp) = εθ(zt) + sg·(εθ(zt,cp) − εθ(zt)).
- SLD 확장: ε‾θ(zt,cp,cS) = εθ(zt) + sg·[εθ(zt,cp) − εθ(zt) − γ(zt,cp,cS)] — 안전 가이던스 항 γ를
  prompt 방향 항에서 빼는 형태(부호 고정, 항상 억제 방향).
- γ(zt,cp,cS) = μ(cp,cS; sS,λ)·(εθ(zt,cS) − εθ(zt)) — 무조건 추정치와 unsafe-concept cS 조건부
  추정치의 차를 억제 방향으로 삼는다.
- μ는 element-wise 게이트: prompt-conditioned와 safety-conditioned 추정치의 차가 threshold λ 미만인
  차원만 스케일 sS로 살리고 나머진 0 — "이미 prompt가 안전 개념 쪽으로 가려는 차원만" 골라 억제
  (SEGA의 sparse dimension-mask와 동일 메커니즘, SLD가 원조).
- warm-up δ(초기 diffusion step엔 γ=0, 구도가 정해진 후에만 개입) + momentum νt(같은 방향으로 계속
  밀리는 차원을 가속) — 이 장치들도 SLD가 먼저 도입, SEGA가 그대로 재사용.
- (Un)safe concept은 고정 텍스트 하나("hate, harassment, violence, suffering, humiliation, harm,
  suicide, sexual, nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use,
  theft, vandalism, weapons, child abuse, brutality, cruelty") — 범용 concept이 아니라 안전이라는
  단일 축에 특화.
- 4단계 설정(Hyp-Weak/Medium/Strong/Max, δ·sS·λ·sm·βm 조합)으로 공격성 조절 — 용도별(전문 작가 vs
  아동용 서비스) 트레이드오프를 사용자가 선택.

## 실험·결과(I2P 부적절 콘텐츠 감소)

- I2P(Inappropriate Image Prompts) 벤치마크가 이 논문의 두 번째 기여: lexica.art 크롤링 실제 사용자
  프롬프트 4703개(hate/harassment/violence/self-harm/sexual/shocking/illegal activity 7개 카테고리),
  Q16+NudeNet 조합 분류기로 자동 평가. 프롬프트의 1.5%만 실제 toxic한데도 부적절 이미지가 생성됨
  (단어 필터링으로는 못 잡음을 시사).
- SD baseline 부적절 확률: 카테고리별 34~52%, overall 39%.
- SLD 적용(overall): Hyp-Weak 29% → Medium 19% → Strong 13% → Max 9%(SD 39% 대비 최대 약 77%
  감소). Negative-prompt baseline(unconditional을 unsafe-concept 조건부로 단순 치환)은 18%로 SLD
  Medium 수준에 그침 — element-wise 게이트(μ)의 fine-grained 억제가 주는 이득을 실증.
- ethnic bias 재현 실험: 일본 75%+ → SLD 적용 후 12.0%(글로벌 평균 9.25%에 근접)이나, SD-unmitigated
  대비 국가간 순위 상관은 유의하게 남음(bias 완화이지 제거 아님).
- 이미지 품질/텍스트 정합 부작용 거의 없음: FID-30k 14.43→18.76(소폭 상승)이나 DrawBench user
  study에서는 사용자가 SLD 결과를 SD와 동등 또는 오히려 선호(60%대) — "안전을 위해 품질을 희생하지
  않는다"는 핵심 주장의 근거.

## §5(산업)에서의 위치(Diffusers opt-in vs 표준 CLIP 필터)

- Diffusers 라이브러리에 `StableDiffusionPipelineSafe`로 정식 탑재(논문 각주 1의 코드 링크가
  `huggingface.co/docs/diffusers/api/pipelines/stable_diffusion_safe`) — `pip install diffusers`로
  누구나 접근 가능하나, 표준 `StableDiffusionPipeline`과는 **별개 클래스**다. 기본 파이프라인을 쓰면
  이 latent steering은 자동 적용되지 않고, 사용자가 명시적으로 Safe 버전을 선택해야 발동한다(opt-in).
- 대비 지점은 SD 자체에 내장된 post-hoc CLIP `safety_checker`다: 생성 "후" 이미지를 분류해 검은
  화면으로 대체하는 방식이며 기본값은 켜져 있으나 코드 한 줄로 끌 수 있고(§동기에서 다룬 lexica.art
  사례), latent 조작 없이 순수 분류기 게이트다. SLD는 반대로 생성 "도중" latent에 개입하지만, 정작
  라이브러리 배포 형태는 default가 아니라 opt-in 대안 파이프라인으로 격하돼 있다 — steering이
  post-hoc 필터보다 기술적으로 더 정교함에도 실무 배포 강도는 더 약하다는 역설.
- 이는 survey §5 교훈("steering의 상용화는 항상-켜진 안전 gate 형태가 지배적")의 반례에 가깝다:
  Circuit Breakers(LoRA로 가중치에 새겨 상시 가동)나 Constitutional Classifiers(실서빙 1단계
  스크리너로 항상 통과)와 달리, SLD는 "표준 라이브러리에 코드가 존재"할 뿐 "기본 경로에서 항상
  실행"되지는 않는다 — SEGA 노트가 지적한 "OSS 표준 탑재 vs 상용 서비스 상시 배치" 중간지대에서 가장
  약한 배포 강도의 사례.

## 우리 프로젝트 연결(¬C_failure 유비)

- SLD의 안전 개념 방향 γ = μ·(εθ(zt,cS) − εθ(zt))는 "부적절 개념 쪽으로 가려는 차원만 골라 반대로
  민다"는 점에서, 우리 C_steer = C_success ∧ ¬C_failure 의 ¬C_failure(실패 방향 억제) 항과 목적이
  정확히 대응한다 — 둘 다 "원치 않는 방향으로의 이동을 감지해 반대로 미는" 억제 연산.
- 다만 연산 형태는 다르다: SLD의 μ는 **차원별 hard gate**(prompt-safe 차가 λ 미만인 좌표만 스케일,
  기저 회전 없음, image score space 위)이고, 우리 conceptor의 ¬C_failure는 **공분산 기반 soft
  subspace 투영**(다차원 aperture로 연속 제어, action-token hidden state 위) — SEGA 노트에서 이미
  다룬 additive-vs-projective·hard-mask-vs-soft-subspace 대비가 SLD에도 그대로 적용된다(SLD가
  SEGA의 안전 특수화이므로 당연).
- 결정적 차이는 **개념 정의 방식과 실패의 성격**이다: SLD의 "부적절 개념"은 고정 텍스트 문자열(정적
  유해 카테고리, 대상이 안정적이고 매 생성마다 동일)이지만, 우리의 "실패"는 rollout마다 동적으로
  발생하는 phase-dependent 사건(길이 confound, task-phase 조건부)이라 텍스트 하나로 방향을 고정할 수
  없다. SLD의 "언제 개입할지(warm-up δ)"는 생성마다 고정된 스케줄인 반면, 우리는 "언제(어느 phase)
  개입할지"를 rollout 진행에 따라 online으로 읽어야 한다 — 이것이 우리 프로젝트의 중심 미해결 문제
  (온라인 phase/failure-type 식별)다.
- SLD의 μ 게이트("prompt가 이미 unsafe 쪽으로 가려는 차원만 억제")는 조건부 개입 아이디어의 원형으로
  볼 수 있다: 우리의 phase-matched steering도 "현재 phase/pathway가 실패 쪽으로 가려는 신호가 있을
  때만" 개입해야 한다는 점에서 구조적으로 닮았다. 다만 SLD는 이 판단을 매 denoising step 다시
  계산되는 로컬 즉석 차이값으로 하고, 우리는 rollout 전체에 걸친 온라인 상태 추적을 필요로 한다는
  시간축 스케일 차이가 있다.

## 면접 포인트(Q→A)

Q1. SLD가 activation/latent steering 계보에서 왜 §5(산업)에 들어가는가?
A. 표준 SD의 post-hoc CLIP `safety_checker`(생성 후 이미지 분류·차단, 대상은 완성된 픽셀)와 달리,
latent space 안에서 생성 도중 "안전하지 않은 개념" 방향으로부터 score를 실시간으로 밀어내는
inference-time steering이기 때문이다. Diffusers 공식 파이프라인(`StableDiffusionPipelineSafe`)으로
배포돼 "이미지 도메인의 안전 gate로서의 activation steering"을 대표하는 사례다.

Q2. SLD와 SEGA의 관계는?
A. 같은 저자군이 같은 CFG 확장 수학을 썼다. SLD(2211.05105, 2022-11)가 먼저 "안전 개념 하나, 항상
억제 방향"으로 좁게 냈고, SEGA(2301.12247, 2023-01)가 "임의 개념, 임의 부호"로 일반화했다 — 형제
논문이며 SLD가 SEGA 수학의 안전 특수 케이스다.

Q3. SLD가 표준 SD safety_checker보다 우월한가?
A. 우월한 축과 열등한 축이 나뉜다. post-hoc CLIP 필터는 이미 생성된 이미지를 사후 검열(검은 화면
대체)하는 반면, SLD는 생성 과정 자체를 부적절하지 않은 이미지로 유도해 콘텐츠 품질(FID/user study)을
유지한 채 부적절 확률을 최대 약 77% 낮춘다는 게 강점이다. 다만 SLD도 Q16/NudeNet 자동분류기의 오차,
hyperparameter 손튜닝, 그리고 결정적으로 **opt-in 배포**(기본 파이프라인이 아님)라는 한계가 있어,
"항상 켜진" 안전장치라는 점에서는 오히려 CLIP checker(기본값 on)보다 약하다.

Q4. 우리 프로젝트에 어떻게 연결되나?
A. ¬C_failure(실패 방향 억제)의 이미지 안전판 유비다. 단 SLD는 정적 텍스트 개념과 고정 warm-up
스케줄로 개입 타이밍을 정하는 반면, 우리는 rollout마다 동적으로 변하는 phase/failure-type을 online
으로 읽어 개입을 라우팅해야 한다는 점에서 훨씬 어려운 문제를 다룬다.

## 한계·비판

- 이론적 근거가 약하다(SEGA와 공유하는 문제): CFG 확장이 실제 classifier gradient와 다르다는 논의
  자체가 본 논문(SLD)엔 없고 후속 SEGA에서야 다뤄진다 — 순수 경험적 방법.
- 자동분류기(Q16/NudeNet) 자체의 오류가 평가 신뢰도를 제한한다: 저자도 "Q16이 보수적이라 Hyp-Max의
  9%가 대부분 false negative로 추정된다"고 인정 — ground truth 없이 분류기로 분류기 성능을 검증하는
  순환성이 있다.
- hyperparameter(δ, sS, λ, sm, βm) 4단계 config가 손튜닝 결과이고 "얼마나 강하게"는 전문 작가 vs
  아동용 서비스 같은 주관적 판단에 의존한다 — 객관적 threshold가 없다.
- ethnic bias는 완화되나 제거되지 않는다: SLD 적용 후에도 국가간 SD-unmitigated 순위와 유의한 상관이
  남아, 학습 데이터 자체의 표현 왜곡을 근본적으로 해결하지 못한다(저자도 별도의 데이터 balancing이
  필요하다고 인정).
- Broader impact에서 저자도 "동일 메커니즘을 반대 부호로 쓰면 부적절 콘텐츠를 오히려 증폭 생성하는
  데도 쓸 수 있다"고 인정한다 — 양날의 검(SEGA와 동일한 경고).
- §5 위치에서 지적한 opt-in 배포 자체가 실무적 한계다: 실제 사용자가 기본 파이프라인을 쓰면 이
  안전장치는 아예 발동하지 않는다 — "항상 켜진 안전 gate"가 되지 못하는 구조적 약점.
