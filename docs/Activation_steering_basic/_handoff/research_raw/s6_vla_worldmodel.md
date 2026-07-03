# §6 VLA / world model steering (신규)

> 주의: 2026 arXiv ID(2602/2603/2606) 다수 → 다운로드 시 반드시 검증. 이름 유사 별개논문 혼동 주의.

| Title | Authors+Year | Venue | arXiv | 관련성 | tier | folder |
|---|---|---|---|---|---|---|
| VLS: Steering Pretrained Robot Policies via VLMs | Liu 2026 | preprint | 2602.03973 | denoising 샘플링을 VLM reward로 steer(학습불필요), 개입지점 비교축 | must | references |
| SAEs Reveal Interpretable & Steerable Features in VLA | Swann 2026 (Stanford) | preprint | 2603.19183 | pi0.5 hidden SAE→motion primitive steering; feature 대부분 memorized(일반화 경고) | must | references |
| Latent Activation Editing (LAE, multirobot nav) | Das 2025 (USC) | preprint | 2509.20623 | online classifier+activation-edit(우리 검출→라우팅과 동형, 도메인 다름) | must | references |
| Do What You Say (runtime reasoning-action align) | Wu 2025/26 (CMU/NVIDIA) | ICRA2026? | 2510.16281 | rollout 재선택+VLM 검증(비-activation online steering 대안) | must | references |
| SteerVLM (VLM lightweight activation steering) | Sivakumar 2025 (VT) | preprint | 2510.26769 | VLM activation steering module, §4→§6 다리 | optional | references |
| DSRL (Steer Diffusion Policy w/ Latent RL) | Wagenmaker 2025 (Berkeley) | preprint | 2506.15799 | latent-noise RL steer(vs conceptor) | optional | references |
| DynaGuide (Steer Diffusion w/ Dynamic Guidance) | Du&Song 2025 (Stanford) | preprint | 2506.13922 | learned dynamics 미래예측으로 denoising guide(phase 조건화) | optional | references |
| Steerable VLA Policies (Hierarchical Control) | W.Chen 2026 (Berkeley/Levine) | RSS2026 | 2602.13193 | 추상화수준별 명령 steer(기존 "Steerable VLAs"와 별개!) | optional | references |
| SteerVLA (Long-Tail Driving) | T.Gao 2026 | preprint | 2602.08440 | 주행 VLA를 VLM reasoning으로 steer(SteerVLM과 별개) | optional | references |
| Chain of World (World Model Thinking in Latent Motion) | Yang 2026 | preprint | 2603.03195 | world model latent를 VLA에 결합, motion disentangle | optional | references |
| What Makes Video WM Latents Action-Relevant | Yeom 2026 (SNU) | preprint | 2606.07687 | WM latent 제어가능성 probe(우리 phase read-out 문제와 평가축 유사) | optional | references |

이미 references/에 있다고 agent가 지적: Event-Grounded SAE for VLA (2605.17204), I-FailSense (2509.16072) — 확인 필요.

흐름: LLM steering→VLA 확장(개입지점 다변화: hidden state[SAE-VLA/LAE]·denoising[VLS]·latent-noise RL[DSRL]·dynamics guide[DynaGuide]·reasoning 재정렬[Do What You Say]). world model은 아직 "steering"보다 "latent가 action-relevant/controllable한가" 검증 초기. 우리=VLA hidden state 다차원 contrastive 연산자 개입 + online phase/type(LAE·DynaGuide와 직접 비교).
