# §5 산업 적용 현황

| 자료 | 출처+연도 | 유형 | URL/arXiv | 관련성 | tier | PDF |
|---|---|---|---|---|---|---|
| Golden Gate Claude | Anthropic 2024 | blog데모 | anthropic.com/news/golden-gate-claude | feature steering 공개 데모 원형 | must | web-only |
| Scaling Monosemanticity / Mapping the Mind | Anthropic 2024 | report | transformer-circuits.pub/2024/scaling-monosemanticity | 프론티어 SAE, 제품 기반기술 | must | web-only |
| Evaluating Feature Steering (bias) | Anthropic 2024 | report | anthropic.com/research/evaluating-feature-steering | steering 실용성 정량평가 | must | web-only |
| Persona Vectors | Anthropic 2025 | report | anthropic.com/research/persona-vectors (arXiv 2507.21509) | persona 모니터·완화 파이프라인 | must | 2507.21509 |
| Claude's Character | Anthropic 2024 | blog | anthropic.com/research/claude-character | persona 설계 배경 | optional | web-only |
| Constitutional Classifiers | Anthropic 2025 | report | arXiv 2501.18837 | 배치된 안전필터 | must | 2501.18837 |
| Next-gen Constitutional Classifiers | Anthropic 2025 | blog | anthropic.com/research/next-generation-constitutional-classifiers | 활성화 프로브 1단계 스크리너(상용) | must | web-only |
| Goodfire Ember | Goodfire 2025 | product | goodfire.ai/blog/announcing-goodfire-ember | steering-as-a-service | must | web-only |
| Understanding & Steering Llama 3 (SAE) | Goodfire 2024 | research | goodfire.ai/research/understanding-and-steering-llama-3 | Ember 전신 | must | web-only |
| Goodfire $50M Series A | Goodfire 2025 | biz | goodfire.ai/blog/announcing-our-50m-series-a | 상용 고객/투자 지형 | optional | web-only |
| Circuit Breakers | Gray Swan+CMU 2024 | paper/product | arXiv 2406.04313 | RepE 안전제품화 | must | 2406.04313 |
| Gemma Scope | DeepMind 2024 | paper/OSS | arXiv 2408.05147 | 오픈 SAE 스위트 | must | 2408.05147 |
| repeng | vgel 2023-24 | OSS lib | github.com/vgel/repeng | grassroots 툴화 | optional | web-only |
| EleutherAI autointerp + sae | EleutherAI 2024 | OSS | blog.eleuther.ai/autointerp | 비영리 인프라 | optional | web-only |

지형: 실제 프로덕션 배치는 소수 — Anthropic(안전 파이프라인 흡수, Constitutional Classifiers 활성화 프로브)·Goodfire(Ember, 단 select-partner 축소)·Gray Swan(circuit breakers). 나머지는 OSS 툴/인프라. steering 상용화는 "안전필터"+"interpretability API" 두 틈새 초기단계. (일부 미래형 arXiv ID는 미확인 → 인용 시 재확인)
