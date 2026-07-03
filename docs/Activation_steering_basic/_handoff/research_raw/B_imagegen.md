# Agent B — 이미지생성 diffusion steering (조사결과)

## 기법 표
| 기법 | 유형 | inf/train | 도구·상용 탑재 | 출처 |
|---|---|---|---|---|
| SEGA (Semantic Guidance) | activation/latent (score에 개념방향 항) | inference, 학습불필요 | **Diffusers 공식 SemanticStableDiffusionPipeline** + A1111 확장(sd-webui-semantic-guidance). 상용서비스 미확인 | 2301.12247 |
| The Stable Artist (SEGA 전신) | activation/latent | inference | SEGA로 흡수 | 2212.06013 |
| Safe Latent Diffusion (SLD) | activation/latent (안전방향 항) | inference | **Diffusers StableDiffusionPipelineSafe(opt-in)**. 표준 SD 기본 safety_checker는 SLD 아님(CLIP post-hoc) | 2211.05105 |
| Asyrp (h-space) | **진짜 activation-space(U-Net bottleneck h 직접수정)=LLM residual steering과 최유사** | Δh 경량넷 사전학습(backbone frozen), 적용 inference | 연구코드만, 도구통합 미확인 | 2210.10960 |
| Concept Sliders | **weight-space LoRA(=steering 아님)** | train LoRA+inference | Civitai 광범위 유통(LoRA 생태계지 activation steering 아님) | 2311.12092 |
| Prompt-to-Prompt | activation(cross-attn map 주입) | inference | Diffusers community pipeline | 2208.01626 |
| Smoothed Energy Guidance (SEG) | activation(self-attn query blur) | inference | **ComfyUI-SEGAttention 노드(서드파티)** | 2408.00760 |
| Semantic-aware CFG (S-CFG) | activation 근접(영역별 CFG) | inference | ComfyUI 커스텀 노드(비공식) | (github shiimizu) |
| SAE steering (DiT) | activation(SAE feature) | SAE 사전학습, base frozen | 연구단계만(ICLR2026 제출) | OpenReview J48XM0au4u |
| (대조) DALL·E 3 캡션개선 | **데이터/학습 개조(steering 아님)** | train | 상용 탑재됨(ChatGPT/DALL·E3) | OpenAI DALL·E3 paper |

## 판정
activation-space steering(SEGA·Asyrp·P2P·SEG)은 **연구로 성숙 + 오픈소스 도구(Diffusers 공식/community, ComfyUI 노드, A1111 확장)엔 실제 탑재**. 그러나 **Midjourney·DALL·E3·Firefly·Ideogram·Leonardo 등 소비자 상용 서비스가 프로덕션에 쓴다는 공개근거 없음(미확인)**. 실제 확인되는 상용 기법(DALL·E3 캡션 재작성)은 train-time 데이터개조지 steering 아님. Concept Sliders는 널리 쓰이나 weight-space LoRA. → "도구 생태계엔 실재, 대중 상용 서비스엔 미확인"이 정직한 결론.

## 다운로드 후보(arXiv): SEGA 2301.12247, SLD 2211.05105, Asyrp 2210.10960, Prompt-to-Prompt 2208.01626, SEG 2408.00760, Concept Sliders 2311.12092
