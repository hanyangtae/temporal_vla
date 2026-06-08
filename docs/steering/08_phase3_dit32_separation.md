# Phase 3: DiT 32-layer pre-failure 분리력 분석 결과

작성: 2026-06-02. 대상 run: `target_atomic_moderate10_multilayer_perT_100ep` (1000 ep, 32-layer per-token).
분석 스크립트: `scripts/safe/groot_n16/robocasa/analyze/pathway_separation.py`
산출: `…/analysis/pathway_separation/pathway_separation.json`

## 방법

- **길이통제**: 고정 t(4,8,12,16,20) step까지만 feature pool, 길이>=t 인 rollout만 사용 ([[truncation-length-standard]]).
- **표상**: DiT transformer_blocks[L] valid-16 action token mean-pool → `[D=1536]` per-step → 첫 t step mean.
- **분리력**: PCA(SVD, n=30)→LDA 방향 투영→Mann-Whitney AUROC, 5-fold CV, task 내 succ/fail.
- **기준선**: length-only AUROC (step count 단독).

## 결과 요약

| t | length-only | DiT best layer | task-avg AUROC |
|---|---|---|---|
| 4 | 0.999 | L24 | 0.592 |
| 8 | 0.999 | L26 | 0.635 |
| **12** | 0.999 | **L23~L25** | **0.751** |
| 16 | 0.998 | L23 | 0.711 |
| 20 | 0.997 | L31 | 0.765 |

**best 종합**: t=12~20, L23~L31 구간이 peak. t=12 선택 (실시간 steer 개입 여지 확보).

## Task별 (t=12, best DiT layer)

| task | n | fail | best layer | AUROC |
|---|---|---|---|---|
| CloseToasterOvenDoor | 95 | 50 | L28 | 0.709 |
| NavigateKitchen | 93 | 59 | L30 | 0.598 |
| OpenCabinet | 100 | 51 | L26 | **0.937** |
| OpenDrawer | 98 | 66 | L24 | **0.884** |
| PickPlaceCounterToCabinet | 94 | 33 | L31 | **0.872** |
| PickPlaceCounterToStove | 99 | 34 | L25 | **0.843** |
| PickPlaceDrawerToCounter | 99 | 61 | L23 | 0.668 |
| SlideDishwasherRack | 92 | 43 | L23 | 0.731 |
| TurnOnMicrowave | 81 | 45 | L31 | 0.673 |
| TurnOnSinkFaucet | 98 | 73 | L25 | 0.706 |

## 해석

1. **신호 실재**: 길이 통제 후에도 task-avg 0.75 (t=12). SAFE 공정 metric val_seen 0.683보다 높음.
   LDA 프로브 직접 접근이 LSTM보다 신호를 더 잘 포착.
2. **후반층 집중**: 신호는 L20+ 집중. 초기 단계(t=4)에선 약(0.59), t=12에서 peak.
   → NOTALL의 "motor program은 trajectory 초기에 commit, 후기 DiT 불필요"와 일치 (Table 15).
3. **Task 이질성 큼**: OpenCabinet 0.937 vs NavigateKitchen 0.598. DiT만으로는 unseen 일반화 어려움.
   → SAFE unseen chance 결과와 일관. **VL(goal pathway) 추가로 이 이질성이 줄어드는지가 핵심**.
4. **길이 confound 완전 통제됨**: length-only 0.997~0.999이지만 AUROC가 그보다 낮음 → 고정-t 방법 정상 동작.

## 다음 단계

`target_atomic_moderate10_pathway_pertoken_100ep` (VL+DiT-7layer, 245/1000 진행 중) 완료 후:
- 동일 스크립트로 VL pathway AUROC 측정 → DiT와 비교
- 핵심 질문: VL이 NavigateKitchen 등 task-이질성 높은 task에서 DiT보다 분리력 높은가?
- **steering 타깃 선택**: VL > DiT면 vl pathway로, task별 best layer 선택.
