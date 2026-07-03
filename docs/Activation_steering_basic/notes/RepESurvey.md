# Taxonomy, Opportunities, and Challenges of Representation Engineering for Large Language Models (Wehner et al., TMLR 2025)

- 출처: Jan Wehner(CISPA), Sahar Abdelnabi(Microsoft), Daniel Tan(UCL — #16 Tan 신뢰성 논문 저자), David Krueger(Mila), Mario Fritz(CISPA) · arXiv:2502.19649 · **TMLR 2025-09 게재(peer-reviewed)** · PDF: docs/Activation_steering_basic/RepESurvey_2502.19649.pdf · 섹션=전체(분야 지도) — §3 framework·§7.2 method 비교·§8.2 평가 best practices·§11.2 메타스터디·§13 challenges 정독 · tier=must(분야 지도) · 한 줄 역할: **activation steering/RepE 분야의 유일한 peer-reviewed 종합 서베이**(>130편) — identification→operationalization→control 3단계 taxonomy로 이 폴더의 모든 논문을 한 좌표계에 올려주는 지도이자, 면접에서 "분야 전체 구조를 그려보라"는 질문의 표준 답안.

## 문제·동기
RepE(activation steering) 논문이 직전 1년에만 100편 이상 쏟아졌는데 체계화(systematization)가 없었다. Mech-interp 서베이(Ferrando, Bereska&Gavves)는 이해(interpretability)에 치우쳐 "제어(control)" 축을 다루지 않는다. 저자들은 3개 seminal 논문(RepE/Zou, ActAdd/Turner, ITI/Li)에서 forward search + 검색어 조합으로 >130편을 수집, 5개 포함 기준(LLM 대상 / concept 표상 식별 / 행동 제어 / post-hoc 식별 / 중간표상 개입)으로 걸러 최초의 전용 서베이를 만들었다. 정의: "**개념에 대한 모델 행동을 제어하기 위해 내부 표상을 조작하는 기법**". 목표 2축 = behavior steering + interpretability.

## 핵심 아이디어 (3단계 pipeline taxonomy)
모든 RepE 방법을 **Representation Identification(RI) → Operationalization → Control(RC)** 파이프라인으로 통일.
- **RI(개념 표상 식별)**: ① **Input Reading** — 대조 입력의 활성화 차이에서 추출(DiM, linear probe, PCA, CCS). ② **Output Optimization** — 원하는 출력이 나오도록 concept operator를 직접 최적화(BiPO 등). ③ **Unsupervised Feature Learning** — SAE로 개념 사전 분해.
- **Operationalization(표상의 수학적 표현)**: 가정 기하(단일 선형 방향 vs **mean&covariance 매칭**) × operator 형태(**vector vs matrix** vs 혼합). conceptor = matrix operator의 대표.
- **RC(개입)**: **활성화 수정**(linear addition / vector rejection / nullspace projection / **soft projection=conceptor** / affine) vs **가중치 수정**(LoRA로 새김, ReFT). 추가 축: 개입 위치(layer/component/token position) + **input-dependent 여부(상시 vs 조건부 on-off vs 강도 조절)** — CAST/ASA류 조건부 steering이 이 마지막 축.

## 방법 비교의 실증 종합 (§7.2 — 면접용 "뭐가 더 잘 듣나")
- **RI 함수 비교: Difference-in-Means가 7편 중 6편에서 최강**. 다음이 linear probe weight, PCA·CCS는 하위. mean-centering(Jorgensen)은 추가 이득 — AxBench(#17)와 독립적으로 수렴하는 결론.
- **operator: matrix가 vector보다 효과적이나 비용↑**(Postmus&Abreu #14·Rajendran·Pham&Nguyen 인용) — 단 출력 품질 저하 사례도 병기. 비선형 operator가 선형보다 나은 증거도 축적 중.
- SAE steering은 결과 엇갈림(이기는 논문 2, 지는 논문 1, AxBench에선 LoReFT·class-means에 밀림) — "SAE = detection엔 좋고 steering엔 애매"가 여기서도 재확인.
- 저자 경고: **"unbiased 비교가 거의 없어 대부분의 우열 결론은 잠정적"**(§14.5) — 제안 논문이 자기 방법에 유리한 비교만 수록.

## 메타스터디 (§11.2, Table 13 — RepE vs 대안 계열)
- **vs prompting**(24편): 제어 효과 RepE 우세 18 / 동률 0 / 열세 7 (75%). **vs fine-tuning**(25편): 18/3/5 (72%), capability 보존도 RepE 우세 8/2/1, **500 샘플 미만에선 RepE가 전승**(sample efficiency). **vs decoding**(7편): 5/0/2 (71%).
- **결합이 최강**: RepE+prompting/FT/decoding 결합을 평가한 4편 전부에서 결합 > 단독.
- **단 저자 스스로 bias 자인**: 비교 대부분이 RepE 논문에서 나와 RepE에 유리하게 편향. AxBench(대규모 중립 평가에선 prompting 우세)를 명시적 반례로 병기 — 이 서베이의 정직성 포인트.

## 평가 best practices (§8.2 — 우리 실험 설계 체크리스트로 쓸 것)
① 강도 스윕 + **음(-)의 강도** 효과 보고(단조 증가면 진짜 그 개념을 조종하는 증거; Tan #16의 steerability metric 인용). ② **capability 변화 보고 — 전체 논문의 ~35%만 함**(RepE의 고질적 보고 누락). ③ OOD 일반화: fit 데이터의 다른 subset이 아니라 **설정 자체를 바꿔** 평가. ④ strong baseline(최신 RepE + 강한 prompting/FT)과 비교. ⑤ MCQ가 아닌 free generation 평가. ⑥ LLM-judge 대신 검증된 specialized judge.

## Challenges (§13 — 우리가 그대로 물려받는 위험 목록)
- **Empirical**: ① **multi-concept 동시 steering 시 간섭·효과 감소**(layer 분산·orthogonality 강제로 완화 시도). ② **long-form/multi-turn에서 제어력 감쇠**(짧은 답으로 fit한 operator가 긴 생성을 못 조종; 동적 강도 조절로 완화). ③ OOD 비일반화(system prompt만 바뀌어도 약화 — Tan). ④ capability 손상(강도에 비례).
- **Principled**: ① **spurious correlation** — DiM/probe는 상관된 개념을 분리 못해 함께 조종됨(faulty code ↔ 일반 비윤리, emergent misalignment 사례). ② **concept misspecification** — 지정한 입력/채점이 의도 개념이 아닌 다른 표상을 활성화("정직해라" 지시가 '인간이 정직을 원한다' 표상을 잡을 수 있음). ③ superposition 간섭(비직교 feature 부작용). ④ 대조 데이터 가용성 가정(비이진 개념엔 대조쌍 구성 불가).
- **적용 범위**: §10.8 "RepE outside LLMs"는 이미지 생성(GAN/diffusion latent 편집)만 다룸 — **로봇/VLA는 부재**(포함 기준 자체가 LLM 한정).

## activation-steering 흐름 위치 (분야의 자기 인식)
이 폴더의 §1~§5 논문들이 각자 낸 주장(DiM 최강, matrix>vector, SAE 애매, brittleness, 조건부 필요)을 **한 좌표계에서 교차 검증한 메타 층위**. RepE(#4)가 top-down 매니페스토라면 이 논문은 그 3년 뒤의 정산서다: "잠재력은 크지만 multi-concept·신뢰성·capability 보존이 미해결"이 공식 결론이고, 우리 서베이 문서(00_activation_steering_survey.md)의 §3 신뢰성 비판·§5 장벽 정리와 사실상 같은 지형도를 그린다(독립 검증). 공저자 구성(Tan=신뢰성 비판 당사자, Krueger=safety)이 비판 축의 신빙성을 더한다.

## 우리 프로젝트 연결
- **우리 방법의 표준 좌표**: 이 taxonomy로 말하면 우리는 "RI=Input Reading(succ/fail 대조, DiM의 다차원 확장) + Operationalization=**matrix operator(conceptor, mean&covariance 기하)** + RC=**soft projection**을 **input-dependent(조건부: pathway·phase 라우팅)**로 적용" — 면접에서 한 문장으로 위치 설명 가능.
- **DiM 6/7 최강** → 우리 사다리 ablation의 "diff-in-means baseline을 반드시 이겨야 한다" 규율이 AxBench에 이어 이 서베이에서도 재확인.
- **multi-concept 간섭 경고** → 우리 multi-layer 동시 steering(all-7-layer SR 0.000 붕괴 관측)과 정확히 같은 패턴 — layer 분산 적용·orthogonality가 문헌의 완화책이라는 것도 참고.
- **long-form 감쇠** → LLM의 긴 생성 ≈ VLA의 긴 rollout. 짧은 구간에서 fit한 operator가 긴 horizon에서 감쇠한다는 관찰은 **phase별 재-fit(phase-matched conceptor)**의 문헌적 근거가 된다.
- **§8.2 best practices** = 우리 실험 보고 체크리스트(강도 스윕·capability(=SR 외 지표) 변화·OOD(=unseen task)·strong baseline)로 그대로 채택 가능.
- **VLA 부재 확인** — >130편 지도에 로봇 정책이 없음 = 우리 niche(내부 latent × online × 실패 type × phase)가 LLM-RepE 정산서 기준으로도 빈자리임을 제3자 문헌이 확인.

## 면접 포인트 (Q→A)
1. Q: "activation steering 분야를 구조적으로 설명해보라." A: "TMLR 2025 RepE 서베이의 3단계 pipeline이 표준 지도다: ① Representation Identification(input reading=대조 활성화 / output optimization / SAE), ② Operationalization(선형 방향 vs 공분산 기하 × vector vs matrix operator), ③ Control(활성화 수정: addition/rejection/projection vs 가중치 수정). 실증 종합은 DiM이 RI 함수 중 최강(7편 중 6), matrix operator가 vector보다 효과적이나 비용↑, SAE는 detection 대비 steering 애매."
2. Q: "RepE가 fine-tuning·prompting보다 낫다는 근거는?" A: "메타스터디(Table 13)에선 제어 효과 기준 prompting 상대 75%, FT 상대 72% 우세, 500샘플 미만에선 FT 전승, capability 보존도 우세. **단** 비교 대부분이 RepE 논문 출신이라 편향됐다고 저자 스스로 경고하고, 중립 대규모 평가(AxBench)에선 prompting이 이긴다 — 그래서 '작은 데이터·켜고 끄기·정밀 제어가 필요할 때 RepE'가 정직한 답."
3. Q: "이 서베이가 꼽는 미해결 난제와 당신 연구의 관계는?" A: "multi-concept 간섭, long-form 감쇠, OOD 비일반화, spurious correlation이 4대 난제다. 내 프로젝트는 이 중 long-form 감쇠(→phase별 conceptor 재-fit)와 조건부 적용(input-dependent control 축의 극단인 online pathway/phase 라우팅)을 VLA에서 직접 다루고, 서베이의 적용 지도(§10.8)에 로봇/VLA가 아예 없다는 것이 그 빈자리의 문헌적 증거다."
4. Q: "steering 실험 보고에서 뭘 꼭 넣어야 하나?" A: "§8.2 기준: 강도 스윕(음의 강도 포함, 단조성 확인), capability 변화(35%만 보고하는 고질 누락), OOD 설정 변화 평가, 최신 strong baseline 비교, free-generation 평가. 우리 ΔSR ladder + held-out seed 분리가 이 체크리스트의 VLA 번역이다."

## 한계·비판
- **LLM 한정**: 포함 기준이 명시적으로 LLM — VLA·diffusion policy·로봇은 §10.8에서 이미지 생성만 스치고 끝. 우리 도메인 질문(rollout 시간축, 물리 실패)에 대한 답은 없다.
- **vote-counting 메타스터디**: 효과크기(effect size) 통합이 아니라 "이긴 논문 수 세기" — publication bias·비교조건 이질성이 그대로 남고, 저자도 bias를 자인한다. "75% 우세"를 강한 주장으로 인용하면 안 됨.
- **스냅샷 시점**: 수집이 2025년 초(arXiv 2502) — 이후의 조건부/agentic 흐름(ASA 등)과 2026년 VLA steering 물결(COAST/NOTALL)은 미반영. VLA 쪽은 우리 서베이 문서 §6~7이 이 지도의 연장선.
- taxonomy가 방법 분류엔 강하나 **"왜 되는가"(§12)는 linear representation hypothesis 요약 수준** — 이론적 깊이는 Park LRH(#3) 원 논문이 더 낫다.
