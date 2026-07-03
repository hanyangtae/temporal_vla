# Agent A — LLM/VLM 배포 현실 (조사결과)

## 사례 표 (핵심)
| 주체 | 무엇 | 읽기/쓰기 | 데모/API/실서비스 | train/inf | 출처 |
|---|---|---|---|---|---|
| Anthropic Golden Gate Claude | SAE feature 증폭 | 쓰기 | **데모(24h 한정)** | inf | anthropic.com/news/golden-gate-claude |
| Anthropic Constitutional Classifiers(원조 2025-02) | 별도 텍스트 분류기 게이팅(ASL-3) | 텍스트분류(활성화 아님) | **실서비스** | 별도 classifier train | anthropic.com/news/activating-asl3-protections |
| Anthropic 차세대 Const. Classifiers(2026-02) | Claude 내부 **활성화 선형 probe**로 1차 스크리닝 | **읽기(probe) 전용** | Sonnet 4.5 실트래픽 shadow deploy(2025-12~2026-01, "production-grade"); 완전 상시화는 문구상 불확정 | inf 읽기 | anthropic.com/research/next-generation-constitutional-classifiers, arXiv 2601.04603 |
| Anthropic Persona Vectors | trait 방향 모니터+훈련중 완화 | 읽기+쓰기(**train단계**) | **연구**(오픈모델만); 감사용 probe는 내부 사용(읽기) | steer=train | anthropic.com/research/persona-vectors |
| Anthropic sycophancy 70~85%↓ | 실배포 개선 | — | 실서비스 | **train(RLHF)+system prompt+classifier**("활성화 조작" 언급 0) | anthropic.com/news/protecting-well-being-of-users |
| OpenAI GPT-4o sycophancy 롤백 | 원인·해결 공개 | — | 실서비스 | **train(RLHF)+prompt**("steer"는 구어) | openai.com/index/sycophancy-in-gpt-4o |
| Google DeepMind Gemini steering(BIDPO) | SAE steering vector 연구 | 쓰기(연구) | **연구전용**; 저자 "conditional steering 있어야 production 가능"=현재 없음 명시 | inf(연구) | turntrout.com/gemini-steering |
| xAI Grok 페르소나 | unhinged comedian 등 | — | 실서비스 | **system prompt(텍스트)**, 활성화 아님 | github.com/xai-org/grok-prompts |
| Character.AI류(Open Character Training) | 페르소나 방법 비교 | 비교연구 | 연구 | **fine-tuning > steering** 결론 | arXiv 2511.01689 |
| Meta Llama Guard | 별도 파인튜닝 텍스트 분류기 | 텍스트분류 | 실서비스(오픈) | 별도모델 train | arXiv 2312.06674 |
| **Goodfire Ember API** | 오픈모델(Llama) feature 실제 조작; jailbreak시 refusal feature 강화(conditional) | **쓰기(steering)** — 가장 명확한 실사용 write 사례 | 유료 API/고객(Rakuten·Apollo·Haize), Series B $150M; **소비자 챗봇 내장인지 백엔드/레드팀인지 불명** | inf | goodfire.ai/blog/announcing-goodfire-ember |
| Transluce Monitor | 자연어로 실시간 feature steer | 읽기+쓰기 | **연구용 오픈도구** | inf | transluce.org/observability-interface |
| EleutherAI SOAR | SAE refusal steering | 쓰기(연구) | 연구; "배포 전 capability tradeoff 해결 필요" | inf | eleuther.ai |
| Gray Swan Circuit Breakers | 유해표현 short-circuit | **train-time(LoRA 가중치)** | 회사는 레드팀 서비스(Cygnal) 실운영; 기법이 프론티어랩 실서빙 이식됐단 근거 미확인 | train | grayswan.ai/research/circuit-breakers |

## 판정
대형 소비자 LLM/VLM 서비스에서 **inference-time activation steering(쓰기)이 서빙 경로에 상시 배치돼 실사용자 응답을 바꾼다는 근거 없음**. 가장 근접 = Anthropic 차세대 Const. Classifiers(내부 활성화 **읽기 probe**를 Claude 실트래픽에 production-grade 운영, 단 읽기지 steer 아님). **쓰기** 실상업사례 유일 = Goodfire Ember(오픈모델 대상 B2B API, 소비자 챗봇 내장 근거 없음). Golden Gate=데모, persona/Gemini=연구, circuit breakers=train-time, Anthropic/OpenAI sycophancy 개선=RLHF/prompt. → **"읽기(probe)는 실서비스에 신중히 진입, 쓰기(steering)는 데모·연구·소수 B2B API 수준"**.

## 다운로드 후보(arXiv): Open Character Training 2511.01689(fine-tuning>steering 근거)
## web-only: Golden Gate, Const. Classifiers(2601.04603/블로그), Goodfire Ember, Gemini steering(turntrout), xAI grok-prompts, Llama Guard 2312.06674, Transluce, Gray Swan
