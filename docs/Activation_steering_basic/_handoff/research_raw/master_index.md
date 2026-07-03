# 마스터 논문 인덱스 (Phase 2 통합)

범례: ★=정독(per-paper agent) 추천 · ◦=스킴(다운로드+짧은노트) · 🔗=web-only(링크만) · ✔=기존 references 확보됨(참조)
folder: B=Activation_steering_basic, R=references

## §1 무엇 & 왜
- ★ Park, Linear Representation Hypothesis & Geometry, 2311.03658, B
- ★ Elhage, Toy Models of Superposition, 2209.10652, B (also §2)
- ★ Zou, Representation Engineering (RepE), 2310.01405, B (also §3/§4)
- ★ Bolukbasi, Debiasing Word Embeddings, 1607.06520, B
- ◦ Mikolov, word2vec, 1301.3781, B
- ◦ K.Li, Othello-GPT (Emergent World Reps), 2210.13382, B
- ◦ Mikolov, Linguistic Regularities (NAACL), N13-1090, 🔗(ACL)
- ◦ Gurnee, LMs Represent Space and Time, 2310.02207, B
- ◦ Engels, Not All Features Are 1-D Linear, 2405.14860, B
- ◦ Huh, Platonic Representation Hypothesis, 2405.07987, B

## §2 Activation 분석 / read-out
- ★ Cunningham, SAEs Find Highly Interpretable Features, 2309.08600, B
- ★ Marks, Geometry of Truth, 2310.06824, B
- ★ Meng, ROME (Locating & Editing Factual), 2202.05262, B
- ★ Wang, IOI Circuit, 2211.00593, B
- ★ Gao, TopK SAE (Scaling & Evaluating SAEs), 2406.04093, B
- ◦ Alain&Bengio, Linear Classifier Probes, 1610.01644, B
- ◦ Belrose, Tuned Lens, 2303.08112, B
- ◦ Burns, CCS (Discovering Latent Knowledge), 2212.03827, B
- ◦ Rajamanoharan, Gated SAE, 2404.16014, B
- ◦ Rajamanoharan, JumpReLU SAE, 2407.14435, B
- ◦ Syed, Attribution Patching, 2310.10348, B
- 🔗 nostalgebraist, Logit Lens (LessWrong)
- 🔗 Bricken, Towards Monosemanticity (transformer-circuits 2023)

## §3 Steering 방법 / write-in
- ★ Postmus&Abreu, Conceptors for Steering, 2410.16314, B  ★★우리 headline
- ★ Turner, Activation Addition (ActAdd), 2308.10248, B
- ★ Rimsky, Contrastive Activation Addition (CAA), 2312.06681, B
- ★ Li, Inference-Time Intervention (ITI), 2306.03341, B
- ★ Todd, Function Vectors, 2310.15213, B
- ★ Wu, ReFT (Representation Finetuning), 2404.03592, B
- ★ Tan, Generalization & Reliability of Steering Vectors, 2407.12404, B
- ★ Wu, AxBench (Simple baselines > SAE), 2501.17148, B
- ◦ Subramani, Extracting Latent Steering Vectors, 2205.05124, B
- ◦ Hendel, ICL Creates Task Vectors, 2310.15916, B
- ◦ Liu, In-Context Vectors (ICV), 2311.06668, B
- 🔗 Mack&Turner, MELBO (LessWrong)

## §4 LLM/VLM 연구
- ★ Arditi, Refusal Is Mediated by a Single Direction, 2406.11717, B
- ★ Sharma, Understanding Sycophancy, 2310.13548, B
- ★ Anthropic, Persona Vectors, 2507.21509, B
- ★ Liu, Reducing Hallucinations in VLM (VTI), 2410.15778, B
- ◦ Gan, Textual Steering Vectors for MLLM, 2505.14071, B
- ◦ Wu, AutoSteer (Safe MLLM), 2507.13255, B
- ◦ Y.Li, VISTA (Hidden Life of Tokens), 2502.03628, B
- ◦ Sivakumar, SteerVLM (VLM activation steering), 2510.26769, B (agent가 R로 뒀으나 VLM이라 B)
- 🔗 Templeton, Scaling Monosemanticity (transformer-circuits 2024)
- 🔗 Anthropic, Golden Gate Claude (blog)

## §5 산업 적용 현황 (대부분 web-only)
- ★ Zou 등, Circuit Breakers (Gray Swan), 2406.04313, B
- ★ Lieberum/DeepMind, Gemma Scope, 2408.05147, B
- ◦ Anthropic, Constitutional Classifiers, 2501.18837, B
- 🔗 Anthropic: Golden Gate / Scaling Monosemanticity / Evaluating Feature Steering / Persona Vectors(research) / Next-gen Constitutional Classifiers
- 🔗 Goodfire: Ember / Understanding & Steering Llama 3 / Series A
- 🔗 vgel repeng, EleutherAI autointerp+sae

## §6 VLA / world model (신규 → R)
- ★ Swann, SAEs Reveal Steerable Features in VLA (pi0.5), 2603.19183, R
- ★ Das, Latent Activation Editing (LAE), 2509.20623, R
- ★ Liu, VLS (Steer Policies via VLM), 2602.03973, R
- ◦ Wu, Do What You Say (reasoning-action align), 2510.16281, R
- ◦ Wagenmaker, DSRL (Diffusion Policy Latent RL), 2506.15799, R
- ◦ Du&Song, DynaGuide, 2506.13922, R
- ◦ W.Chen, Steerable VLA Policies (Hierarchical), 2602.13193, R
- ◦ T.Gao, SteerVLA (Driving), 2602.08440, R
- ◦ Yang, Chain of World, 2603.03195, R
- ◦ Yeom, Video WM Latents Action-Relevant, 2606.07687, R
- ✔ 확보됨: NOTALL, COAST, SAFE, Observing&Controlling, Steerable VLAs(InSight), Event-Grounded SAE(2605.17204), dr_vla, CoT-VLA, Scaling World Model

## §7 VLA 산업 적용 방향 (신규 → R)
- ★ Agia, Sentinel (Runtime Monitoring), 2410.04640, R
- ★ Römer, FIPER (Failure Prediction at Runtime), 2510.09459, R
- ★ Ren, KnowNo (Robots That Ask For Help), 2307.01928, R
- ★ Fei, LIBERO-Plus (Robustness Analysis), 2510.13626, R
- ◦ Lin, FailSafe (Recovery in VLA), 2510.01642, R
- ◦ Seo, Uncertainty-aware Latent Safety Filters, 2505.00779, R
- ◦ Zhou, Code-as-Monitor, 2412.04455, R
- ◦ Dai, See Plan Rewind, 2603.09292, R
- ◦ Feldman, Robot Safety from Sparse Feedback, 2501.04823, R
- ◦ Kim, Modular Safety Guardrails, 2602.04056, R
- ◦ Zhang, Benchmarking VLA, 2511.11298, R
- ✔ 확보됨: I-FailSense(2509.16072), PathDeviationHeads(2603.13782), VITA

## 집계
- ★ 정독 추천: 29편 (B 20 + R 9)
- ◦ 스킴: 27편
- 🔗 web-only: ~12
- ✔ 확보됨(참조): 12
- ⚠ 2026 arXiv ID(2602/2603/2606 등) 다운로드 시 검증 필수

## 추가 (2026-07-02, 검증 세션)
- ★ 정독 #52: **RepE Survey** — Wehner/Abdelnabi/Tan/Krueger/Fritz, "Taxonomy, Opportunities, and
  Challenges of Representation Engineering for LLMs", arXiv 2502.19649, **TMLR 2025-09 (peer-reviewed)**.
  선정 이유: 51편 체계에 유일하게 빠져 있던 "분야 지도(서베이)" 슬롯. 대안 Bartoszcze 2502.17601
  (Survey and Research Challenges)은 미게재 preprint라 web-only로 분류.
  PDF=`../../RepESurvey_2502.19649.pdf`, note=`../../notes/RepESurvey.md`.
