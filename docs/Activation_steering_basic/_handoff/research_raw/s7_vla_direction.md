# §7 VLA 산업 적용 방향 (실패검출·복구·안전)

> 주의: 2025~2026 arXiv ID 다수는 다운로드 시 검증 필요(미래형 ID 있음).

## (a) online 실패 검출/모니터링
| Title | Authors+Year | Venue | arXiv | 관련성 | tier | folder |
|---|---|---|---|---|---|---|
| Sentinel (Unpacking Failure Modes of Generative Policies) | Agia 2024 | CoRL2024 | 2410.04640 | 불일치(STAC)+진행정체 2축 실시간 검출 = 우리 phase/type 원형 | must | references |
| FIPER (Failure Prediction at Runtime) | Römer 2025 | NeurIPS2025 | 2510.09459 | embedding OOD+chunk entropy 조기 실패예측(latent) | must | references |
| Code-as-Monitor | Zhou 2024 | CVPR2025 | 2412.04455 | VLM 코드 제약검사(symbolic 대안) | optional | references |

## (b) 복구/재계획/introspection/TTA
| KnowNo (Robots That Ask For Help) | Ren 2023 | RSS2023 | 2307.01928 | conformal 불확실성 정렬, "도움요청" 시조 | must | references |
| FailSafe (Reasoning & Recovery in VLA) | Lin 2025 | preprint | 2510.01642 | VLA plug-in 실패추론+복구 | must | references |
| See, Plan, Rewind (Progress-Aware VLA) | Dai 2026 | preprint | 2603.09292 | progress 추적+rewind 복구(phase 겹침) | optional | references |

## (c) 안전 제어/guardrail
| Uncertainty-aware Latent Safety Filters | Seo 2025 | preprint | 2505.00779 | world model latent OOD conformal 필터(steering 최근접) | must | references |
| Learning Robot Safety from Sparse Human Feedback | Feldman 2025 | preprint | 2501.04823 | latent 위험영역 conformal 식별 | optional | references |
| Modular Safety Guardrails for FM Robots | Kim 2026 | preprint | 2602.04056 | 재학습없는 runtime safety envelope 포지션 | optional | references |

## (d) 평가/신뢰성
| LIBERO-Plus (Robustness Analysis of VLA) | Fei 2025 | preprint | 2510.13626 | 7축 섭동 취약성 정량(SR 붕괴) | must | references |
| Benchmarking VLA for Manipulation | Zhang 2025 | preprint | 2511.11298 | 3축 표준비교, failure taxonomy | optional | references |

흐름: 검출(Sentinel/FIPER)→복구(KnowNo/FailSafe/Rewind)→안전필터(latent conformal, steering 최근접)→신뢰성 벤치(LIBERO-Plus). 우리 phase-matched steering = 요청/재계획 없이 latent 내 즉시 복구하는 저비용 대안 정당화.
