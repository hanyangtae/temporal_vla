# §4 LLM/VLM steering 연구

| Title | Authors+Year | Venue | arXiv/URL | 관련성 | tier | folder |
|---|---|---|---|---|---|---|
| Refusal Is Mediated by a Single Direction | Arditi 2024 | NeurIPS2024 | 2406.11717 | 거부=단일방향, ablation=jailbreak | must | basic |
| Representation Engineering (RepE) | Zou 2023 | NeurIPS2023-WS | 2310.01405 | 다속성 안전 제어 프레임(중복 §1) | must | basic |
| Inference-Time Intervention (ITI) | Li 2023 | NeurIPS2023 | 2306.03341 | attention head truthful 주입 | must | basic |
| Activation Addition (ActAdd) | Turner 2023 | preprint | 2308.10248 | 최적화 없는 steering 원조(중복 §3) | must | basic |
| Contrastive Activation Addition (CAA) | Rimsky 2023/24 | ACL2024 | 2312.06681 | 대비쌍 steering 기준방법(중복 §3) | must | basic |
| Understanding Sycophancy in LMs | Sharma/Perez 2023 | ICLR2024 | 2310.13548 | 아첨 원인 분석(개입 근거) | must | basic |
| Scaling Monosemanticity (Claude 3) | Templeton 2024 | TCT(blog) | transformer-circuits.pub/2024/scaling-monosemanticity | 대규모 SAE feature clamping | must | basic |
| Golden Gate Claude | Anthropic 2024 | blog | anthropic.com/news/golden-gate-claude | 화제 steering 데모(web-only) | optional | basic |
| Persona Vectors | Anthropic 2025 | preprint | 2507.21509 | persona trait 추출·모니터·완화 | must | basic |
| Reducing Hallucinations in VLM via Latent Steering (VTI) | Liu 2024 | preprint | 2410.15778 | LVLM 환각 억제(멀티모달) | must | basic |
| Textual Steering Vectors Improve Visual Understanding (MLLM) | Gan 2025 | preprint | 2505.14071 | cross-modal steering 전이 | must | basic |
| Automating Steering for Safe MLLMs (AutoSteer) | Wu 2025 | EMNLP2025 | 2507.13255 | MLLM 안전 자동 개입 | optional | basic |
| Hidden Life of Tokens (VISTA) | Y.Li 2025 | ICAIC2025 | 2502.03628 | 시각정보 steering 환각억제 | optional | basic |

흐름: RepE/ITI/ActAdd(정직성 방향)→CAA/sycophancy(행동축)→Arditi refusal(안전=단일방향)→Scaling Monosemanticity/Golden Gate(SAE feature 제어)→Persona Vectors→VLM 이식(VTI/VISTA/AutoSteer/Textual)→VLA 다리.
