# §2 Activation 분석 / Read-out

| Title | Authors+Year | Venue | arXiv/URL | 관련성 | tier | folder |
|---|---|---|---|---|---|---|
| Linear Classifier Probes | Alain&Bengio 2016 | ICLR2017-WS | 1610.01644 | probing 원조 | must | basic |
| CCS (Discovering Latent Knowledge) | Burns 2022 | ICLR2023 | 2212.03827 | 비지도 probing, 인과성 논쟁 | must | basic |
| Logit Lens | nostalgebraist 2020 | LessWrong(blog) | lesswrong AcKRB8wDpdaN6v6ru | lens 시초(web-only) | must | basic |
| Tuned Lens | Belrose 2023 | preprint | 2303.08112 | lens 발전형(affine probe) | must | basic |
| Geometry of Truth | Marks/Tegmark 2023 | preprint | 2310.06824 | PCA/mass-mean 방향추출+causal | must | basic |
| Toy Models of Superposition | Elhage 2022 | TCT | 2209.10652 | SAE 동기(중복 §1) | must | basic |
| Towards Monosemanticity | Bricken 2023 (Anthropic) | TCT | transformer-circuits.pub/2023/monosemantic-features | SAE 최초 성공 | must | basic |
| Scaling Monosemanticity (Claude 3) | Templeton 2024 | TCT | transformer-circuits.pub/2024/scaling-monosemanticity | 실규모 SAE(중복 §4) | must | basic |
| SAEs Find Highly Interpretable Features | Cunningham 2023 | ICLR2024 | 2309.08600 | 학계 독립검증 SAE | must | basic |
| Scaling & Evaluating SAEs (TopK) | Gao 2024 (OpenAI) | preprint | 2406.04093 | TopK, 스케일링 | must | basic |
| Gated SAE | Rajamanoharan 2024 (DeepMind) | preprint | 2404.16014 | shrinkage 해결 | optional | basic |
| JumpReLU SAE | Rajamanoharan 2024 | preprint | 2407.14435 | SOTA reconstruction | optional | basic |
| ROME (Locating & Editing Factual) | Meng 2022 | NeurIPS2022 | 2202.05262 | causal tracing 원조 | must | basic |
| IOI Circuit (Interpretability in the Wild) | Wang 2022 | ICLR2023 | 2211.00593 | activation patching 대표 | must | basic |
| Attribution Patching | Syed 2023 | NeurIPS2024-WS | 2310.10348 | patching 스케일링 | optional | basic |

흐름: probe/lens(상관)→CCS/mass-mean(방향추출)→SAE(superposition 해소)→activation patching(인과 검증)→§3 write-in 다리.
