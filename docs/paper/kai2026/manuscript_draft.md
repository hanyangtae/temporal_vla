# VLA 내부 활성화에 나타나는 세밀 행동 단계 구조

(제7회 한국 인공지능 학술대회 투고 초안 — 최종본 최대 2쪽. 저자 목록·소속은 사용자
확정 사항. 수치는 전부 `numbers.md` 대조표 경유, 서지는 `references.md` 경유.
그림: Fig.1 = figs/fig1_purity_residual, Fig.2 = figs/fig2_granularity_margin.)

## 초록

Vision-Language-Action(VLA) 모델이 로봇 조작을 수행하는 동안 내부 활성화가 어떤 시간
구조를 갖는지 조사한다. GR00T N1.5의 action head(DiT) 잔차 활성화를 라벨 없이
군집화하면, 사람이 정의한 primitive 행동 단계(reach, grasp 등)보다 3~4배 세밀한 상태
열이 나타난다(평균 구간 길이 4.3 vs 16.1 스텝). 각 군집은 특정 행동 단계에 국한되어
나타나며(순도 0.80), 시간 진행도 대조군보다 1.67 bits 높은 단계 정보를 담는다. 이
정보는 장면별 평균을 제거한 잔차에서도 유지되어 장면 암기로 환원되지 않는다. 이
구조는 군집 모델의 구조(AE↔SAE)를 바꿔도 같은 시점에서 전환하고(z +8.8~+13.5), 다른
수집 데이터에서도 재현된다(z +4.3~+5.1). 즉 VLA 활성화에는 사람의 단계 정의보다
세밀하지만 단계 정합적인 하위 구조가 실재하며, 스텝 단위 온라인 판독이 가능하다.

## 1. 서론

로봇 조작 태스크는 통상 reach → grasp → transport 같은 시간적으로 확장된 primitive
단계(phase)로 기술된다[9]. VLA 모델의 내부 활성화가 이런 단계 구조를 담고 있는지,
담고 있다면 어떤 해상도로 담는지는 열려 있는 질문이다. VLA 내부 표현을 해석하려는
최근 시도들은 희소 오토인코더(SAE)로 residual stream에서 파지·이송 같은 운동
primitive에 대응하는 feature를 찾아내거나[3,5], 선형 probe로 층별 상태를 관측하고 그
방향으로 활성화를 밀어 행동을 제어할 수 있음을 보였다[4,6]. 그러나 이들은 발견한
feature를 사람이 rollout 영상과 수동으로 정렬해 이름 붙일 뿐, 그 표현이 사람이 정의한
행동 단계와 어떤 해상도로 대응하는지를 체계적으로 정량화하지 않는다. 한편 라벨 없이
조작 궤적을 분절하려는 연구는 시연 데이터의 시각 특징을 군집화하는 방식으로
이어지지만[8], 분절의 근거를 정책 바깥의 관측에서 찾는다 — 정책이 스스로 무엇을 하고
있는지가 내부 활성화에 어떤 단위로 적혀 있는지는 별개의 질문이다. 내부 단계를 추론
중에 읽을 수 있으면 단계 조건부 개입[6]이나 실패 감지[7]의 조건 신호로 쓸 수 있다.
본 연구는 GR00T N1.5(GR00T N1[1]의 후속 릴리스)가 RoboCasa[2] 주방 조작을 수행하는
동안의 action head 활성화를 라벨 없는 군집화만으로 분석하고, 사람이 붙인 단계
라벨(GT)과의 관계를 시간 대조군 대비 margin으로 정량화한다.

## 2. 방법

**데이터·특징.** RoboCasa[2] 주방 조작 9개 태스크에서 GR00T N1.5 rollout을 수집하고,
action head(DiT) 잔차 스트림 layer 12, denoising step 3의 49개 토큰 활성화를 평균해
스텝당 1536차원 특징을 얻는다. 사람 주석 GT 단계 라벨(태스크당 3~6단계)을 평가에만
사용한다. 두 데이터셋을 사용한다: (A) 23 에피소드(단계 경계 91개), (B) 독립 재수집
930 에피소드(10개 태스크, 89,766 스텝; 그중 1개 태스크는 30 에피소드 소표본).

**군집화.** PCA-64(whitening) 후 KMeans로 스텝별 활성화를 군집화한다(k=24; 태스크별
분석은 k=8). 오토인코더 등 학습형 압축기는 이득이 오차 범위 내라 생략한다. train
split에서만 군집을 만들고 평가 split은 k-NN(k=15)으로 라벨을 옮긴다 — 재군집화가
없으므로 추론 중 스텝 단위 온라인 판독이 가능하다.

**지표.** 군집열이 담는 단계 정보는 MI(군집; GT 단계)로 재되, k가 커지면 MI가 자동
상승하므로 **시간 대조군 margin**을 판정 기준으로 쓴다: 에피소드 진행도를 발견 군집
수와 같은 개수의 분위로 자른 "시계(clock)" 대조군의 MI를 빼준 값이다. 경계 정렬은
전환 위치를 에피소드 내에서 무작위로 흩은 300회 대비 z-score로 잰다.

## 3. 결과

**활성화 상태는 GT 단계보다 3~4배 세밀하다(Fig.2a).** 발견 상태의 평균 구간 길이는
4.3 스텝으로 GT 단계(16.1 스텝)의 약 1/4이다(전환율 18.4% vs 5.0%; 데이터셋 B에서도
6.0 vs 18.1 스텝, 전환 수 3.5배). 이 전환은 고정 리듬 아티팩트가 아니고(전환 위치의
주기성 없음), 전환 시점이 GT 경계와 정렬되지도 않는다(z −1.0~+0.5 — 우연 수준).
그러나 발견 상태를 GT 단계 단위로 병합하면 경계 정렬이 유의해진다(z +1.3~+4.9). 즉
활성화는 GT 단계를 그대로 재현하는 것이 아니라 그보다 잘게 쪼갠다.

**각 군집은 특정 단계에 국한된다(Fig.1a, Fig.2b).** 한 단계에서 나타난 군집은 다른
단계에서 잘 나타나지 않는다: 군집→단계 순도 0.80, MI 2.17 bits(상한 2.82), 시간
대조군 margin +1.665 bits. 군집이 최빈 단계 밖에서 나타나는 비율(off-phase rate)의
군집별 중앙값은 태스크 중앙값 0.22이며, 순수한 태스크에서는 0.001~0.06까지
내려간다(넓게 걸치는 태스크도 있어 최대 0.47 — 태스크별 편차가 있다). 단계 하나는
평균 2~3개 군집으로 세분되어(단계당 군집 수 1.3~2.7), 대응은 1:1이 아니라 다대일이다
— 이것이 위의 "세밀한 하위 분할"과 순도가 공존하는 이유다.

**단계 정보는 장면 암기로 환원되지 않는다(Fig.1b).** 활성화에는 장면(scene) 정보도
섞여 있다(태스크별 mi_scene 0.2~1.0 bits). 그러나 태스크 내 장면별 평균을 제거한
잔차에서 장면 정보는 대부분 사라지는 반면(mi_scene 중앙값 0.54→0.24 bits) 단계
margin은 소표본 퇴화 태스크 1개를 제외한 9개 태스크 전부에서 양수로 유지된다
(중앙값 +0.34→+0.28; 3개 태스크는 오히려 상승).

**구조는 모델·데이터를 바꿔도 재현된다.** 같은 데이터에서 구조가 다른 두 압축 모델
(AE, SAE)이 같은 스텝에서 전환하고(F1 0.66~0.75, 무작위 0.51~0.53, z +8.8~+13.5),
독립 재수집 데이터에서 새로 학습해도 같은 성질이 나타난다(seed 간 F1 0.55~0.58,
z +4.3~+5.1). denoising 슬롯·장면·시드는 에피소드 내 상수라 전환의 원인이 될 수 없다.

**세밀 단위는 하류 분석에서도 유용하다.** 성공/실패 분리도를 단계 조건부로 잴 때,
발견 상태(태스크별 k=8)가 GT 단계보다 높은 분리도를 주는 태스크가 9개 중 4개였다
(동급 1, 열세 1; 예: OpenDrawer AUROC 0.93/z4.5 vs GT 0.94/z2.3). 이는 탐색적
관찰이나, 세밀 단위가 사람 정의 단계의 단순 잡음이 아님을 시사한다.

## 4. 한계와 결론

GT 라벨 두 벌은 같은 주석 체계의 두 버전이라 "어떤 단계 정의와도 어긋난다"는 주장은
할 수 없고, 전환 경계 자체는 GT와 정렬되지 않으므로 단계 판독의 근거는 군집 정체성
(MI·순도)이지 경계 검출이 아니다. 길이 1의 짧은 상태가 구간의 31~37%를 차지하며 그
의미는 미검증이다.

결론: VLA 활성화에는 사람의 primitive 단계 정의보다 세밀하지만 단계 정합적인 상태
구조가 재현 가능하게 존재하며, k-NN 라벨 전이로 추론 중 온라인 판독이 가능하다. 이
구조는 단계 조건부 개입[6]과 실패 감지[7]의 조건 신호로 쓸 수 있다.

## 참고문헌

(정본 서지·확인 경로는 `references.md`. 번호는 본문 인용 번호와 일치.)

[1] NVIDIA GEAR Team, "GR00T N1: An Open Foundation Model for Generalist Humanoid
Robots," arXiv:2503.14734, 2025.
[2] S. Nasiriany et al., "RoboCasa: Large-Scale Simulation of Everyday Tasks for
Generalist Robots," RSS, 2024.
[3] A. Swann et al., "Sparse Autoencoders Reveal Interpretable and Steerable Features
in VLA Models," arXiv:2603.19183, 2026.
[4] H. Buurmeijer et al., "Observing and Controlling Features in Vision-Language-Action
Models," arXiv:2603.05487, 2026.
[5] X. Jin et al., "Event-Grounded Sparse Autoencoders for Vision-Language-Action
Policies," arXiv:2605.17204, 2026.
[6] B. Häon et al., "Mechanistic Interpretability for Steering Vision-Language-Action
Models," CoRL, 2025.
[7] Q. Gu et al., "SAFE: Multitask Failure Detection for Vision-Language-Action Models,"
NeurIPS, 2025.
[8] W. Wan et al., "LOTUS: Continual Imitation Learning for Robot Manipulation Through
Unsupervised Skill Discovery," ICRA, 2024.
[9] R. S. Sutton, D. Precup, and S. Singh, "Between MDPs and Semi-MDPs: A Framework for
Temporal Abstraction in Reinforcement Learning," Artificial Intelligence, 112:181–211,
1999.
