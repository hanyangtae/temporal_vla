# SAE 3논문 통합 + 우리 설계 (방향 2)

개별 노트: [dr_vla_sae.md](dr_vla_sae.md), [event_grounded_sae.md](event_grounded_sae.md), [observing_controlling.md](observing_controlling.md).
목적: activation에서 **outcome(succ/fail) 성분을 scene/task·길이 confound와 분리**해 steer. 배경 =
겉보기 DiT succ/fail 분리가 confound라는 결과([[dit-succfail-apparent-separation-confound]]).

## 1. 세 논문 핵심 (한 줄씩)

- **Dr.VLA** (Stanford, 2603.19183): TopK+AuxK SAE(1× expansion, block residual, PaliGemma **L5**), per-sample normalize. 핵심 기여 = **generality-vs-memorization metric**(rollout·라벨 없이 4개 활성화 통계 = coverage/onset/magnitude/**relative-run-length** + logistic). feature의 89~99.5%가 "암기". steer = 단일 decoder-column 덧셈(정성 인과만, ΔSR 없음).
- **Event-Grounded SAE** (Purdue, 2605.17204): 이름과 달리 **event-conditioned SAE가 아님** — 평범한 BatchTopK SAE + **event 기반 feature 순위선택(post-hoc)**. event = AWE keyframe 클러스터, feature를 pulse/step 템플릿과 **모양(크기 아님)** 매칭해 랭킹. OpenVLA L31만 event-정렬 zero-out ΔSR −21 유의; **π0.5 action-expert(AE)는 아무 랭킹에나 붕괴=비선택적** → **GR00T DiT가 바로 이 AE 유형**(위험 신호).
- **Observing & Controlling** (Stanford/NVIDIA, 2603.05487): **SAE 아님**. 선형 observer(Wx+b, pose/gripper) + 최소 덧셈 control u=(ζ_t−ζ)W/‖W‖². 얕을수록 controllable(단 norm-growth 아티팩트). flow/diffusion head steering은 **명시적 보류**(=우리 DiT 타깃을 그들도 안 함). arbiter = closed-loop SR(우리 ΔSR 철학과 동일).

## 2. 수렴하는 결론 (셋을 겹쳐 보면)

1. **어느 논문도 outcome-vs-scene/길이 분리를 직접 풀지 않음.** 두 SAE 논문 다 **같은 scene/task confound가 통제 안 된 채** 존재(EG-SAE success probe: SAE code 0.79–0.93 vs task-id-only 0.54–0.64 = task 누출). Observing은 dense kinematic 라벨이라 애초에 outcome를 안 다룸.
2. **공통 처방 = phase(event) bin 안에서 대조 conceptor fit** → 길이/phase confound를 구조적으로 통제. **우리 fit 어댑터가 이미 per-phase group으로 이걸 함**(방향 1). EG-SAE도 "phase bin 안에서 C_succ∧¬C_fail"을 우리에게 권고.
3. **DiT ≈ π0.5 AE = 단일 feature steer에 붕괴/비선택적.** → **단일 SAE feature를 DiT에 steer하지 말 것. 다차원 conceptor(우리 메인 method)를 써야 함.** 세 논문이 우리 conceptor 선택을 역으로 정당화.
4. **SAE의 우리에게 진짜 쓸모(Dr.VLA)**: 우리 scene confound(AUROC 1.0) = 그들의 "**암기 feature**". SAE가 **scene/task를 별도 sparse feature로 분리**해 주면, **scene-free 부분공간에서 outcome를 찾고 거기서 conceptor fit** 가능. 그들의 metric 축이 우리 confound와 정확히 대응: **relative-run-length = 우리 길이축, onset = phase축**.
5. **"event-grounded" = SAE 구조가 아니라 feature 랭킹 휴리스틱** → 우리에겐 "SAE feature를 grasp/place/release로 랭킹/선택"으로 번역(무료지만 검정력은 안 늘림).
6. **Observing의 이식 포인트**: 선형 **progress/phase observer**(dense 라벨→표본풍부·저분산)를 online phase 신호원으로(경량 VITA 대체), 그리고 rank-1 최소 control을 **가장 싼 ΔSR baseline rung**으로.
7. **셋 다 못 고치는 것 = 우리 3~4 fail/scene 저검정력.** phase-conditioning은 confound만 줄이지 표본을 안 늘림. 더 많은 실패 또는 scene-pool 설계가 여전히 필요.

## 3. 우리 SAE 빌드 설계 (phased, 사다리 규칙)

전제: **방향 1(raw residual 다차원 conceptor) steered ΔSR가 먼저**. 그게 신호를 보이면 SAE는 "정제"로, 안 보이면 SAE는 "scene 분리 후 재시도"로 — 어느 쪽이든 방향1 인과 결과가 SAE 착수를 정당화(사다리: raw가 말한 뒤 복잡도 추가).

- **S1 — SAE 학습 인프라**: TopK/BatchTopK per-token SAE(1× expansion, k≈64), tap = 우리 캡처 DiT residual `[7,4,1536]`(AE류; K=4 denoise 유지) + VL `[2048]`(PG류). 학습 데이터 = 기존 rollout 활성화(hook 인프라 재사용). dead-latent AuxK.
- **S2 — confound feature 식별**: Dr.VLA 4-metric(coverage/onset/**run-length**/magnitude) + 각 feature를 **scene(seed)/길이/phase/outcome에 회귀** → scene·길이·phase 설명 feature 제거 → **잔여 outcome-특이 feature** 후보.
- **S3 — scene-free 공간에서 conceptor + 인과**: 남은 부분공간에서 **phase별 대조 conceptor** fit → residual-preserving edit(x' = x + Dec(z')−Dec(z))로 steer → **ΔSR 인과 판정**. 단일 feature 아니라 다차원(§2-3).
- **caveat(정직)**: (a) 저검정력은 SAE로 안 풀림 → 실패 표본 보강 병행, (b) DiT 붕괴 위험 → 다차원 필수, (c) predictive≠steerable(flow 비선형; Dr.VLA 경고), (d) Eagle-LM≠PaliGemma라 layer는 새로 sweep, (e) 라이선스(Dr.VLA repo) 확인.

## 4. 미점유 niche (우리 기여 후보)

세 논문 모두 **outcome↔feature 지도학습 회귀(scene/길이 통제 포함)**를 SAE 위에 얹지 않음. Dr.VLA metric은 generality용, EG-SAE 랭킹은 event용 — **succ/fail을 confound 통제 하에 직접 타깃하는 SAE-feature 선택은 빈자리**([[notall-online-failuretype-niche]] 연장). 이게 우리 method 후보.

## 5. 빌드 스펙 (repo-grounded, 실행 가능)

정찰(2026-07-01): 기존 SAE/AE 코드 **없음**(net-new). 재사용 가능 = `src/conceptor`(S3), `src/ttt/progress_head.py`(Observing 선형 observer/progress analog), `scripts/serve/safe_hooks.py` tap.

**데이터 — 신규 수집 불필요.** 이미 캡처된 `phase_event_aligned_4cell/raw_rollouts` **6 cell × 15 = 90 rollout** pkl 이 SAE 학습셋:
- 입력 = DiT block residual `[L=7, K=4, D=1536]`(layer [0,2,4,8,10,12,15]) + VL `[2048]` per record. `analyze/phase_separation.py:load_rollout` 재사용.
- **라벨이 전부 동봉** → S2 회귀에 바로 사용: **scene**=cell_id(seed), **length/progress**=rollout 내 record-index, **phase**=feature_phases, **outcome**=episode_success.
- 규모: layer당 ~7k record 벡터(K pool 시), K=4를 sample로 펴면 ~28k. 1× SAE엔 얇음 → capacity 보수적 + 실패 보강 병행(caveat). 참고 규모: Dr.VLA LIBERO 273k timestep, Event-Grounded SAE당 ~500 rollout(per-token) → **우리가 ~10–40배 적음 → SAE 착수 전 데이터 스케일업(scene·rollout↑) 필요**(실패 표본 보강과 같은 방향).
- **★ 하드 규칙(사용자 지시): 학습 단위 = per-record(=per-timestep), phase 라벨 per-record 유지. rollout 전체를 1벡터로 뭉개는 pooling(episode_mean류) 절대 금지** — action phase가 timestep별로 구분되는 게 load-bearing이라 rollout-pool은 phase 구조를 파괴한다. K=4 denoise 축 pool은 시간축과 직교라 허용(단 per-timestep 유지). [[feedback-no-rollout-pooling]]

**tap (재사용).** SAE 학습·추론 주입 지점 = safe_hooks.py 의 `transformer_blocks[ℓ]` residual(DiT motor) / `action_head.vlln`(VL goal). 추론 SAE-steering hook 은 `ConceptorSteering` 패턴 그대로.

**S1 — TopK SAE 학습 (net-new).**
- `src/sae/topk_sae.py`: TopK+AuxK(Gao), 1× expansion(dict=입력차원), k≈64, per-sample normalize(median pre-bias→mean→l2), unit-norm decoder, dead-latent(500 step) AuxK. 손실 = ‖x−x̂‖² + (1/32)AuxK.
- `scripts/safe/groot_n15/robocasa/sae/train_sae.py`: 90 pkl→활성화셋→학습→ckpt. **열린결정**: tap layer(conceptor overlap 좋은 L4/L8 후보), K 처리(pool vs final-denoise-step; Dr.VLA=final).
- env=torch(lerobot_safe 또는 원격 anaconda).

**S2 — confound feature 식별 (net-new, 우리 niche).**
- `sae/analyze_sae_features.py`: 각 SAE feature에 Dr.VLA 4-metric(coverage/onset/**run-length**/magnitude) + **scene/length/phase/outcome 회귀**(로지스틱·선형). scene·length·phase 설명 feature 마스킹 → **잔여 outcome-특이 feature** 후보. (미점유 niche = outcome 직접 타깃 + scene 통제.)

**S3 — scene-free 공간 conceptor + 인과.**
- outcome-특이 부분공간에서 **phase별 대조 conceptor**(`src/conceptor` 재사용) → residual-preserving edit `x'=x+Dec(z')−Dec(z)` steer hook → **ΔSR**(`steer/steer_eval_30x30.sh` 재사용). **단일 feature 금지, 다차원**(DiT 붕괴 회피, §2-3).

**순서 게이트(사다리).** 방향1 raw conceptor ΔSR 결과 후 착수 — SR↑면 SAE=정제(일반성·해석), 무효면 SAE=scene 분리 후 재시도. **열린 결정 종합**: tap layer/K, expansion(1× 시작), scene 통제 방식(회귀 잔차화 vs within-scene vs orthogonal), 저검정력(SAE로 안 풀림→실패 보강), 라이선스(Dr.VLA repo).
