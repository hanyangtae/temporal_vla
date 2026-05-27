# RoboMD: Uncovering Robot Vulnerabilities through Semantic Potential Fields

> 워크숍 버전 제목: *"From Mystery to Mastery: Failure Diagnosis for Improving Manipulation Policies"* (RSS 2025 Workshop OOD).
> arXiv 최신본(v4)에서 제목이 **"RoboMD: Uncovering Robot Vulnerabilities through Semantic Potential Fields"** 로 개정됨.

## 메타 정보
- **arXiv ID**: 2412.02818 (최신 v4, 2026-05-18 업로드)
- **저자**: Som Sagar, Jiafei Duan, Sreevishakh Vasudevan, Yifan Zhou, Heni Ben Amor, Dieter Fox, Ransalu Senanayake
- **소속**: arXiv abstract 페이지에 미표기 (명시되지 않음)
- **게재 venue**: RSS 2025 Workshop OOD (워크숍 제목 "From Mystery to Mastery", Published 2025-06-01). 메인 컨퍼런스 채택 여부는 명시되지 않음.
- **게재 날짜**: 최초 제출 2024-12-03, 최신본 v4 2026-05-18
- **프로젝트 페이지**: 명시되지 않음 (abstract 페이지에 링크 없음)
- **코드 공개 여부**: 명시되지 않음 (abstract 페이지에 코드 링크 없음)

## 한 줄 요약 (1 sentence)
성공/실패 라벨로 학습한 비전-언어(VL) 임베딩 공간을 "semantic potential field"로 간주하고, 그 위에서 deep RL 에이전트(PPO)가 가상 롤아웃으로 정책이 취약한 환경 변형(failure-prone variation)을 탐색·랭킹하는 **정책-무관(policy-agnostic) 취약점 진단 프레임워크**.

## 문제 정의
manipulation 정책이 어떤 환경 조건(색상, 조명, 물체 속성 등)에서 실패하는지를 체계적으로 파악하기 어렵다는 문제를 다룬다. 모든 변형을 무차별 테스트하는 것은 비싸고, 특히 학습 분포 밖(OOD)의 실패는 사전에 발견하기 더 어렵다. RoboMD는 제한된 성공-실패 데이터만으로 정책의 취약점을 예측하고, 이를 통해 타겟 fine-tuning에 필요한 데이터량을 줄이면서 manipulation 성능을 개선하는 것을 목표로 한다.

## 방법 (Method)
- **핵심 아이디어**: ViT(이미지) + CLIP(태스크 텍스트) dual-backbone으로 성공/실패 롤아웃을 512-d VL 임베딩 공간 ℰ로 사상한다. BCE + contrastive loss(Eq.1)로 "같은 결과(성공/실패)는 가깝게, 다른 결과는 멀게" 구조화하여, 임베딩 자체가 potential function Φ(s)=e_s 역할을 하도록 만든다. 그 위에서 PPO 에이전트 π_MD가 potential-based reward shaping F(s,a,s')=γΦ(s')−Φ(s)를 사용해 failure cluster 쪽으로 탐색을 유도한다.
- **입력 / 출력**: 입력 = 이미지 + 태스크 텍스트 + 성공/실패 라벨(롤아웃). 출력 = 환경 변형별 실패 확률 맵("probabilistic vulnerability-likelihood map") 및 취약점 랭킹.
- **모델 아키텍처**: dual-backbone(ViT + CLIP) 임베딩 인코더 + PPO 정책 π_MD.
- **학습 / 추론 단계 구분**: 둘 다 존재. π_MD는 임베딩 공간 내 **가상 롤아웃**으로 학습(물리 로봇 불필요). 학습 후엔 임의 시나리오의 실패 여부를 **쿼리**(추론)하여 seen/unseen 조건을 랭킹.
- **실패 정의 / 발견 방식**: 이진 성공/실패 결과를 **환경 변형**에 연결. categorical 분류도, 순수 OOD 탐지도 아님 — 분포 내·외 실패를 모두 다룸("characterize failures both within and beyond the training distribution"). **연속** 임베딩 공간에서 RL 탐색으로 unseen 변형까지 발견. 완전 unsupervised는 아니며 라벨 데이터(벤치마크에서 500 pair)가 필요.
- **작동 공간**: **VL 임베딩 level**. raw policy latent도, action level도 아님. RL 에이전트의 action a_t ∈ ℝ^512가 임베딩 벡터이며, ℰ_known 내 nearest neighbor로 실제 적용될 환경 변형을 결정.
- **Pseudo code (Algorithm 1, Sec 3.3)**:
  ```
  Input: Dataset 𝒟 = {(x_i^vision, x_i^lang), y_i}
  Precompute ℰ_known from 𝒟
  For t = 0 to N:
    Sample action a_{t+1} ~ π_MD(s_t)
    a_{t+1}* ← nearest neighbor in ℰ_known
    s_{t+1} ← transition using a_{t+1}*
    r ← r + R(s_{t+1})     # Eq.2: failure region 유도 + ℰ_known 이탈/반복 행동 페널티
  ```

## 실험
- **사용한 모델(진단 대상 정책)**: (sim) Behavior Cloning, Hierarchical BC, BC-Transformer, BCQ, Diffusion. (real) ModAttn(UR5e). (ablation) 최신 VLA로 π0, SmolVLA, GR00T 언급 — v4에서 추가된 것으로 보이며, 보고된 내용 그대로 기재.
- **벤치마크**: RoboSuite(lift, stack, threading, pick & place) — Robomimic/MimicGen 데이터셋 기반. 실로봇 UR5e.
- **Baseline 비교 대상**: RL 알고리즘(A2C, SAC, PPO), VLM(GPT-4o, Gemini 1.5 Pro, Qwen2-VL), small model(CNN, ResNet).
- **핵심 수치**:
  - 취약점 랭킹 정확도: PPO 평균 80.7% vs GPT-4o-ICL 54.3%.
  - 일반화: unseen item 랭킹 61–80%.
  - Fine-tuning 효과: 타겟 실패 데이터로 92.83% (vs pre-trained 67.91%), 필요 데이터 **1.3 GB targeted** vs **9.0 GB all failures**.
  - VLM baseline 대비 최대 **23% 더 많은 unique vulnerability** 발견.
- **Ablation**: image+text(BCE+Contrastive)가 image-only/BCE-only보다 우월(MSE 0.1801). 임베딩 품질(silhouette 0.84–0.95, 낮은 Davies-Bouldin). 조명 sweep에서 단조 거리 증가(Kendall's τ=1.0)로 semantic continuity 확인.

## 우리 wedge와의 정확한 차이
우리 wedge:
- **Frozen VLA action expert residual stream(z_mean 1024-d)에서의 unsupervised failure mode discovery**
- **Mode-conditional conceptor steering at inference time**
- **Latent 기반 auto-labeling (VLM 기반 labeling baseline과 대비)**

1. **Failure mode 분해 여부 + 방법**: RoboMD는 실패를 *환경 변형*(색/조명/객체 속성) 단위로 랭킹한다 — 즉 입력 조건(environment-side) cluster를 분해한다. mode를 나누긴 하지만 그 대상은 "정책 내부의 실패 행동 mode"가 아니라 "외부 환경 요인"이다. 우리는 정책 내부 latent에서 실패 *행동* mode를 비지도로 분해한다.
2. **작동 공간**: RoboMD = 별도로 학습한 VL semantic 임베딩(512-d, ViT+CLIP). 우리 = frozen VLA의 action expert residual stream(1024-d 정책 내부 latent). 정책 외부 semantic vs 정책 내부 latent.
3. **Training-time vs Inference-time**: RoboMD는 진단/랭킹용 RL을 학습하고, 실제 개선은 *타겟 fine-tuning(정책 가중치 재학습)* 으로 이뤄진다. 우리는 정책 재학습 없이 inference-time steering으로 개입한다 — RoboMD가 "진단→재학습" 루프라면, 우리는 "진단→추론시 직접 개입".
4. **Mode-conditional intervention 유무**: RoboMD의 "steering"은 RL 탐색을 failure region으로 *유도*(진단 목적)하는 것이지, 배포된 정책의 행동을 실시간으로 교정하는 것이 아니다. 개선은 fine-tuning 데이터 선별이라는 간접 경로다. 우리는 mode별 conceptor로 정책 출력을 직접 조건부 교정한다.
5. **VLA에 적용 가능한가**: RoboMD는 policy-agnostic 진단 도구로 VLA에도 적용 가능(v4에서 π0/GR00T 등 언급). 단 frozen latent 개입이 아니라 외부 진단이므로, 우리와 같은 family라기보다 **보완재**다.

## Reviewer 대응 시나리오
> "이건 RoboMD와 같지 않냐?"

"RoboMD는 *어떤 환경 조건이 실패를 유발하는지*를 외부 VL 임베딩 위에서 랭킹하는 진단 도구이며, 개선은 타겟 fine-tuning(정책 재학습)에 의존한다. 우리는 frozen VLA의 action latent에서 실패 *mode*를 비지도로 분해하고, 재학습 없이 mode-conditional conceptor로 추론시에 정책을 직접 교정한다 — 진단 대상(외부 환경 vs 정책 내부 latent)과 개입 방식(재학습 vs inference steering)이 모두 다르다."

## 핵심 인용 포인트
- **Related Work**: "failure diagnosis / vulnerability discovery" 단락 — 성공-실패 라벨로 학습한 임베딩에서 실패 요인을 탐색하는 계열의 대표로 인용. 우리의 latent 기반 auto-labeling을 "VL-embedding 기반 외부 진단"과 대비.
- **Experiments (baseline/비교)**: RoboMD가 GPT-4o 등 VLM보다 취약점 발견에서 우월하다고 보인 점을 인용해 "VLM 라벨링의 한계"를 강조하면서, 우리는 한 발 더 나아가 VLM 없이 latent auto-labeling으로 가능함을 주장.
- **Discussion**: "진단→재학습" 루프 대비 "진단→inference steering"의 데이터 효율/배포 단순성 논의.

## 한계 / 우리에게 유리한 점
- **RoboMD 한계**(본문에 명시적 limitation 절은 없음 — 암묵적 제약): 라벨 성공-실패 데이터 필요, "proximity = outcome similarity" 가정에 의존, 가설적 환경 변형의 물리적 구현 가능성 전제, 고차원 action space/멀티태스크 확장 미논의. 개선이 fine-tuning(정책 가중치 변경)을 요구.
- **우리에게 유리**: 우리는 frozen 정책 + 재학습 없음으로 배포·데이터 효율 측면에서 유리. 또한 RoboMD는 정책 내부 행동 mode를 보지 않으므로 "같은 실패 trajectory를 반복(loop)"하는 정책 내적 실패를 직접 다루지 않는다 — 이는 우리 핵심 문제의식과 직교/보완 관계.
