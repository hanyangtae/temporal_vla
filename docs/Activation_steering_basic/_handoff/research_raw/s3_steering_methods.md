# §3 Steering 방법 / write-in

| Title | Authors+Year | Venue | arXiv/URL | 기법군·관련성 | tier | folder |
|---|---|---|---|---|---|---|
| Activation Addition (ActAdd) | Turner 2023 | preprint | 2308.10248 | additive/single/boost 원조 | must | basic |
| Contrastive Activation Addition (CAA) | Rimsky 2023/24 | ACL2024 | 2312.06681 | additive mean-diff 표준 baseline | must | basic |
| Inference-Time Intervention (ITI) | Li 2023 | NeurIPS2023 | 2306.03341 | additive, head별 subspace | must | basic |
| Representation Engineering (RepE) | Zou 2023 | preprint | 2310.01405 | control 우산 프레이밍 | must | basic |
| Extracting Latent Steering Vectors | Subramani 2022 | ACL2022-F | 2205.05124 | learned(최적화) 원조 | optional | basic |
| Function Vectors | Todd 2023 | ICLR2024 | 2310.15213 | function/task vector 대표 | must | basic |
| ICL Creates Task Vectors | Hendel 2023 | EMNLP2023-F | 2310.15916 | task vector 이론 | optional | basic |
| In-Context Vectors (ICV) | Liu 2023 | ICML2024 | 2311.06668 | task vector 실적용 | optional | basic |
| Conceptors for Steering | Postmus&Abreu 2024 | NeurIPS2024-WS | 2410.16314 | **projective/subspace, C_succ∧¬C_fail 직접근거** | must | basic |
| Refusal = Single Direction | Arditi 2024 | NeurIPS2024 | 2406.11717 | projective(directional ablation) suppression | must | basic |
| ReFT (Representation Finetuning) | Wu 2024 | NeurIPS2024 | 2404.03592 | 경량 학습 intervention(LoReFT) | must | basic |
| Generalization & Reliability of Steering Vectors | Tan 2024 | ICML2024 | 2407.12404 | **신뢰성 한계(언제 실패)** | must | basic |
| AxBench (Simple baselines > SAE) | Wu 2025 | ICLR2025 | 2501.17148 | **비판 벤치(diff-in-means 강함)** | must | basic |
| MELBO (unsupervised latent behaviors) | Mack&Turner 2024 | LessWrong | web-only | 비지도 steering 탐색 | optional | basic |

흐름: additive(ActAdd→CAA→ITI)→task/function vector→projective/subspace(conceptor·directional ablation)→RepE 통합→ReFT(학습형)→신뢰성 비판(Tan·AxBench, diff-in-means가 오히려 강함 → 우리 phase/pathway 조건부 steering 근거).
