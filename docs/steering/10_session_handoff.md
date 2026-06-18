# 세션 핸드오프 — latent steering 연구 현황

작성: 2026-06-05. 다음 세션이 cold-pickup 없이 이어갈 수 있도록 작성.

---

## 연구 목적 (한 줄)

GR00T N1.6 RoboCasa에서 실패를 latent 신호로 사전 감지하고, 그 타입(goal/motor)에 맞는
pathway를 conceptor steering해서 SR을 올린다. 이게 되지 않더라도 failure-type 자동 분류를
달성하는 것이 최소 deliverable.

> 갱신(2026-06): main stream은 **pathway 분리 + phase-matched steering**으로 구체화됨. 핵심
> 난제 = **online phase/failure-type 식별**. 현재 방향 단일 출처: `14_pathway_phase_online_steering.md`.

---

## 왜 이 방향인가 — 이전 실패와 NOTALL 분석

**이전 두 시도 모두 실패:**
- SAFE detector (DiT pre-velocity feature, LSTM): val_unseen AUROC 0.43 (chance)
- COAST conceptor steering (DiT 32-layer): 평균 ΔSR ≤ 0 (모든 조건에서 음수)

**NOTALL(ICLR 2026, `docs/references/NOTALL.txt`) 이 실패 이유를 설명:**
- 두 시도 모두 **DiT(motor "how") pathway만** 사용했음
- NOTALL 핵심: GR00T에는 세 pathway — DiT(motor), Eagle-VLM(goal "what"), VL-SA(bridge)
- **VLM pathway가 goal semantics 인코딩, DiT는 motor program** → failure type에 따라 어느 pathway를 봐야 하는지가 다름
- GR00T N1.6에서 **VL pathway = `action_head.vlln`** (post-LayerNorm, D=2048, get_action당 1회 발화, K=4 denoising 전파)

---

## 이번 세션에서 한 실험들

### Phase 1: VL+DiT 동시 capture/steering 인프라 구축

**왜:** 기존 코드는 DiT-only. VL pathway hook이 없어서 goal failure 신호를 볼 수 없었음.

**무엇:**
- `scripts/serve/steering_hooks.py`: `ConceptorSteering`에 `pathway∈{dit,vl}` 추가
  - `pathway="vl"`: `action_head.vlln` hook, 전체 VL token steer (`h' = h·Mᵀ`)
  - `pathway="dit"`: 기존 DiT block hook (backward compatible)
- `scripts/safe/groot_n16/robocasa/serve/feature_server.py`: `--capture-vl`, `--steering-pathway` 추가
- collect client/artifacts: `vl_hidden_states` per-step pkl 저장
- Launcher: `scripts/safe/groot_n16/robocasa/steer/collect_pathway_parallel.sh` (trap cleanup, port-scoped kill)

**검증:** 단위테스트 8/8, smoke 1-ep: VL[2048]+DiT[7,51,1536] 한 forward에서 정렬 확인.

---

### Phase 2: VL+DiT 정렬 데이터 수집

**왜:** 기존 perT run(1000 pkl, 32-layer)은 inference_seed 미고정 → 재현 불가, VL 없음 → 재활용 불가.
새 run은 seed 고정 + ep_meta 저장 → 재현 가능, 나중에 추가 layer 수집도 aligned로 가능.

**데이터:**
- Run ID: `target_atomic_moderate10_pathway_pertoken_100ep`
- moderate-10 task × 100 ep = 1000 pkl
- DiT 7-layer subset (0,2,4,8,16,24,31) per-token + VL seq-mean-pool
- pkl schema: `hidden_states[step] = [7, 51, 1536]`, `vl_hidden_states[step] = [2048]`
- **원격 보관**: `kimseungjun@166.104.146.37:11112` `/home/kimseungjun/workspace/temporal_vla/outputs/eval/robocasa/groot_n16/target_atomic_moderate10_pathway_pertoken_100ep/raw_rollouts/` (34GB)
- 로컬: raw_rollouts 삭제됨. `analysis/`, `ep_meta/` 보존.

---

### Phase 3: Pathway 분리력 비교 (cv-AUROC)

**왜:** DiT가 실패를 얼마나 잡는지, VL이 추가로 얼마나 더 잡는지 측정. steering 타깃 선택 근거.

**방법:** 고정-t inference step 평균 feature → PCA(SVD) → LDA → Mann-Whitney AUROC, 5-fold CV.
길이 통제 필수 (실패=항상 45 inference step timeout, 성공=조기종료 → length-only AUROC=0.999).

**스크립트:** `scripts/safe/groot_n16/robocasa/analyze/pathway_separation.py`

**결과 파일:**
- DiT-32 run: `outputs/.../target_atomic_moderate10_multilayer_perT_100ep/analysis/pathway_separation/pathway_separation.json`
- VL+DiT-7 run: `outputs/.../target_atomic_moderate10_pathway_pertoken_100ep/analysis/pathway_separation/pathway_separation.json`

**결과 요약 문서:** `docs/steering/08_phase3_dit32_separation.md`, `docs/steering/09_phase3_vl_dit_comparison.md`

**핵심 발견:**

| t | VL(goal) | DiT-b31(motor) | 해석 |
|---|---|---|---|
| t=4 | **0.677** | 0.648 | VL이 더 이른 신호 |
| t=8 | **0.713** | 0.701 | VL 우위 |
| t=12 | 0.745 | **0.752** | 동등 |
| t=20 | 0.741 | 0.743 | 동등 |

Task별 분열:
- **VL 우위**: SlideDishwasherRack(0.931), CloseToasterOvenDoor(0.800) → goal-type failure
- **DiT 우위**: OpenCabinet(0.912), OpenDrawer(0.888), PnP tasks(0.87~0.91) → motor-type failure
- **둘 다 미유의**: NavigateKitchen, PickPlaceDrawerToCounter, TurnOnSinkFaucet

NOTALL 예측 재현: VL=goal semantics, DiT=motor program 전문화가 RoboCasa에서도 확인됨.

---

### LDA 기반 failure quadrant 분류

**왜:** Phase 3의 scalar Mahalanobis는 1D로 압축해서 정보 손실. LDA direction을 직접 써야
AUROC=0.931을 만든 실제 discriminant를 따라 anomaly를 측정함.

**스크립트:** `scripts/safe/groot_n16/robocasa/analyze/vl_dit_lda_analysis.py`

**결과 파일 (task별):**
```
outputs/.../target_atomic_moderate10_pathway_pertoken_100ep/analysis/lda_quadrant/
  SlideDishwasherRack/
    lda_scatter.png        ← LDA-VL vs LDA-DiT 산점도
    lda_trajectory.png     ← step별 LDA anomaly (사분면별 평균)
    lda_per_quad.png       ← 사분면별 대표 ep 4개 trajectory
    video_list.html        ← 카테고리별 비디오 목록 (watch_video.sh 명령어 포함)
    lda_summary.json       ← 사분면별 ep 목록 + score
  OpenCabinet/
    (동일 구조)
```

**사분면 정의:**
- A (VL+ DiT+): 둘 다 혼란 — 복합 실패
- B (VL+ DiT-): goal-type failure 가설
- C (VL- DiT+): motor-type failure 가설
- D (VL- DiT-): 둘 다 정상인데 실패 — latent로 예측 불가

**SlideDishwasherRack 결과:** A=14, B=8, C=8, D=14 (고르게 분포 → 분류 의미있음)
**OpenCabinet 결과:** A=20, B=4, C=4, D=19 (A/D에 몰림 → 두 pathway가 함께 움직여 분리 약)

**비디오 보는 법:**
```bash
! bash scripts/safe/groot_n16/robocasa/watch_video.sh task7--ep53--succ0.mp4 SlideDishwasherRack
```
실행 후 workspace 루트에 mp4 복사됨 → IDE 파일 탐색기에서 클릭 재생.

---

## 다음 세션이 해야 할 것

### 우선순위 1: B/C 비디오 검증 (정성 확인)

`video_list.html` 열어서 SlideDishwasherRack B그룹(goal-type, 주황)과 C그룹(motor-type, 파랑)
영상을 실제로 보고 "goal 오인인가 vs 동작 실패인가"를 확인. 가설이 맞으면 논문 근거가 됨.

추천 순서: B그룹 ep53, ep38 → C그룹 ep91, ep62 → 성공 ep 1~2개 비교.

### 우선순위 2: Phase 4 — VL conceptor fit + SR eval

`fit_conceptor_steering.py`가 현재 DiT-only. VL pathway 지원 추가 필요:
- `--pathway vl` 옵션 추가 → `action_head.vlln` feature를 rollout에서 로드
- fit 후 `eval_steer_compare.sh`를 VL pathway로 확장
- 실험 매트릭스: VL β∈{0.1, 0.3} × always-on/online × 7개 task (미유의 3개 제외)
- 핵심 비교: type-matched steer가 unmatched보다 ΔSR 높은가

### 우선순위 3: perT run 정리

다른 세션에서 COAST Phase D(layer_select_compare) 완료 확인 후:
```bash
# 검증 후 삭제
rm -rf outputs/eval/robocasa/groot_n16/target_atomic_moderate10_multilayer_perT_100ep/raw_rollouts
```

---

## 주요 파일 위치 요약

| 목적 | 경로 |
|---|---|
| 연구 방향 | `docs/steering/09_phase3_vl_dit_comparison.md` |
| 방법론 survey | `docs/steering/07_steering_methods_survey.md` |
| DiT 분석 결과 | `docs/steering/08_phase3_dit32_separation.md` |
| VL vs DiT 비교 | `docs/steering/09_phase3_vl_dit_comparison.md` |
| pathway hook | `scripts/serve/steering_hooks.py` |
| feature server | `scripts/safe/groot_n16/robocasa/serve/feature_server.py` |
| 수집 launcher | `scripts/safe/groot_n16/robocasa/steer/collect_pathway_parallel.sh` |
| 분리력 분석 | `scripts/safe/groot_n16/robocasa/analyze/pathway_separation.py` |
| LDA 사분면 분석 | `scripts/safe/groot_n16/robocasa/analyze/vl_dit_lda_analysis.py` |
| 비디오 다운로드 | `scripts/safe/groot_n16/robocasa/watch_video.sh` |
| 분리력 결과 JSON | `outputs/.../target_atomic_moderate10_pathway_pertoken_100ep/analysis/pathway_separation/pathway_separation.json` |
| LDA 사분면 결과 | `outputs/.../target_atomic_moderate10_pathway_pertoken_100ep/analysis/lda_quadrant/{task}/lda_summary.json` |
| 비디오 목록 HTML | `outputs/.../analysis/lda_quadrant/{task}/video_list.html` |
