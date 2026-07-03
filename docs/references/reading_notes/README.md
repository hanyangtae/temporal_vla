# reading_notes — 방향 2 (SAE) 논문 학습·설계 index

**목적**: activation에서 **outcome(succ/fail)을 scene/task·길이 confound와 분리**해 steer하는 방법 탐색.
배경 = 겉보기 DiT succ/fail 분리가 confound라는 실측 결과([[dit-succfail-apparent-separation-confound]]).

## 문서

| 문서 | 한 줄 |
|---|---|
| [SAE_synthesis_and_design.md](SAE_synthesis_and_design.md) | **먼저 볼 것** — 3논문 통합 + 수렴 결론 + repo-grounded 빌드 스펙(S1~S3) |
| [dr_vla_sae.md](dr_vla_sae.md) | Dr.VLA: TopK+AuxK SAE(PaliGemma L5), **generality-vs-memorization metric**(라벨 없이 4통계). 우리 scene confound=그들 "암기 feature" |
| [event_grounded_sae.md](event_grounded_sae.md) | Event-Grounded SAE: **event-conditioned 아님** = 평범 BatchTopK + event 기반 feature 랭킹. residual-preserving edit. **AE(=우리 DiT)는 비선택적 붕괴** |
| [observing_controlling.md](observing_controlling.md) | **SAE 아님** = 선형 observer+최소 control. flow head steering 보류. arbiter=SR. → progress observer(src/ttt) + 최싼 ΔSR baseline |

## 수렴 결론 (3논문 겹침)

1. 어느 것도 outcome-vs-scene/길이 분리를 직접 안 풂 (두 SAE 다 scene/task 누출 통제 안 됨).
2. 공통 처방 = **phase bin 안에서 대조 conceptor** (우리 fit 어댑터가 이미 함).
3. **GR00T DiT ≈ π0.5 AE = 단일 feature steer 붕괴** → 다차원 conceptor 필수(우리 메인 정당).
4. SAE 쓸모 = scene(암기) feature 분리 → scene-free 공간에서 outcome + conceptor.
5. 셋 다 3~4 fail/scene 저검정력 못 고침 → 실패 보강 병행.
6. 미점유 niche = succ/fail 직접 타깃 + scene 통제 SAE-feature 선택.

## 빌드 순서·게이트

방향1 raw conceptor **ΔSR 결과 후 착수**(사다리). → S1 TopK SAE(기존 90 rollout pkl 학습, 신규수집 불필요)
→ S2 confound feature 회귀 제거 → S3 scene-free conceptor + residual-preserving steer → ΔSR.
상세 = SAE_synthesis_and_design.md §5. 통합 메모리 = [[sae-study-synthesis]].
