# exp5-2 핸드오프 — 섭동-유도 실패의 activation steering 회복 실험

작성 2026-07-27 (exp4-2 세션 종료 핸드오프). 이 문서만 읽으면 시작 가능하도록 자기완결로
작성. 근거 상세는 [`28_exp4-2_p0_report.md`](28_exp4-2_p0_report.md) (특히 §9–§11),
인프라 runbook 은 `scripts/safe/groot_n15/robocasa/steer/induced/README.md` (**함정 목록
필독** — bash GROUPS 변수, kitchen lang 재생성, wrapper private 접근, docker exec -i 등).

## 0. 목표 (사용자 확정, 07-27)

**"SR 을 깎는 섭동(perturbation)을 activation steering 으로 회복할 수 있는가"** — WAM
(arXiv:2607.14943, `24c_walqr_reference.md`) 식 교란-하 ΔSR 이 본선. 자연실패와의 정렬은
부차(기술 통계로만). 대상 섭동 = **C1(카메라)·P1(displace)·P2(force)** — G1 은 분리 없음
으로 제외(아래 §1). **task+instruction 은 3종으로 제한: ppcc_bread / drawer left / mixer.**

| cell | task | env | cell_index | seed | instruction |
|---|---|---|---|---|---|
| ppcc_bread | PickPlaceCounterToCabinet | robocasa_panda_omron/PickPlaceCounterToCabinet_PandaOmron_Env | 5 | 100084 | Pick the bread from the counter and place it in the cabinet. |
| drawer left | OpenDrawer | robocasa_panda_omron/OpenDrawer_PandaOmron_Env | 8 | 100001 | Open the left drawer. |
| mixer | OpenStandMixerHead | robocasa_panda_omron/OpenStandMixerHead_PandaOmron_Env | (27번 문서) | feasibility 통과 seed | (instruction 1종 — docs/steering/27) |

## 1. 확정 결과 — 어디서(layer), 어떤 분리가 (ppcc_bread, scene 100084)

측정: clean 짝 = `baseline_cap`(같은 seed capture 재수집, P1 trigger 이전 clean↔perturbed
**bitwise 동일 실측** — state-matched 증명). held-out AUROC(방향 짝수 ep fit→홀수 평가),
d′ = 평균이동/그 방향 표준편차, 누출 = clean 90% PCA 부분공간 밖 초과 잔차.

**평균(1차) 분리 — 주 신호**:

| | L0–L4 | L8 | L10 | L12 | L15 | VL |
|---|---|---|---|---|---|---|
| C1 AUROC / d′ | .70–.78 / .3 | .96 / .82 | .99 / .99 | **1.00 / 1.09** | .97 / .75 | **.99 / 1.80** |
| P1 AUROC / d′ | .87–.91 / .3 | .98 / .55 | **.98 / .64** | .98 / .63 | .97 / .48 | .91 / .96 |
| P2 AUROC / d′ | .83–.86 / .5 | .96 / .74 | .97 / .92 | **.99 / .99** | .98 / .98 | **.98 / 1.49** |
| (G1 제외 근거) | .70–.73 | .79 | .80 | .79 | .82 | .75 |

**분산(2차) 구조 변화**:
- 유효차원(participation ratio 비): L12–15 에서 1.4–1.7배 확장 (전 모드), L8 이하는 ≈1.1.
- 부분공간 초과 누출: **L8 까지 ≈0–3% (같은 공간 안의 이동)** → L12 5–10%, L15 7–14%,
  VL 12–22% (공간 이탈 시작).
- 섭동 효과의 rank: paired Δh top-1 EVR — **P1 .77–.91, P2 .69–.84 (거의 단일 방향)**,
  C1 .33–.60 (확산형, 다차원).

**종합**: 분리는 **L8 에서 형성돼 L10–L12 가 정점**. L8–L12 는 "분리는 크고 공간은 아직
공유"라 steer-back 의 기하학적 적지. P1/P2 는 rank-1 방향으로 조준 가능, **C1 은 VL 이
최강(d′ 1.80)이고 다차원** — C1 은 VL 개입이 정합.

## 2. Steering 처방 (pre-register 후 실행할 것)

**연산자** (명명 규약: setpoint mean-diff = **setM**, 위약 = **setM_pl**, 다차원 = conceptor
— 축약 신조어 금지, memory feedback-arm-naming-no-abbrev):
1. **setM (primary, P1·P2·C1-DiT)**: r̂ = normalize(μ_clean − μ_perturbed) (주의: 회복 방향
   = perturbed→clean 이므로 clean 이 양의 목표), s = μ_clean·r̂, h′ = h − β[(h·r̂)−s]·r̂.
   exp4-1 이 구현한 **setM affine hook** 배선 재사용 (dev 병합분 — `--steering-*` 벡터 NPZ
   키 `alpha0_v_steer`/`alpha0_s`, 24a §4.1). fit 은 반드시 **per-episode 균등 서브샘플
   k=20**(dwell 통제 — 07-23 실증: 미적용 시 방향 최대 ~30° 회전).
2. **conceptor (C1-VL·대조)**: 기존 fit 파이프라인 (`fit_phase_conceptor_n15.py --ep-subsample-k 20
   --record-start-manifest ...`), C_steer = C_clean ∧ ¬C_perturbed. VL 개입은 serve 의 VL
   steering 경로(`--steering-pathway vl`) 확인 필요 — DiT 는 기성.
3. **위약 setM_pl**: 라벨 순열 fit, held-out ‖Δh‖ 분포 일치 확인 후 동결 (dose-matched).

**개입 layer**: DiT **L8·L10·L12** (사다리 — 단계별로만 확장), C1 추가로 **VL**.
**개입 시점**: C1 은 전 구간(지속 섭동), P1/P2 는 **trigger 이후 latch** (trigger record 는
스펙에 있음 — client `--gated-steering` + `/steering_phase` 배선 기성, exp4-1 latch 참조).

**평가 프로토콜 (WAM 식, 자기완결 — t0 주석 불필요)**:
- 같은 (scene, inference_seed, spec_seed) 섭동 rollout 에서 **steering ON vs OFF vs setM_pl**
  paired 비교. Primary contrast = ON vs setM_pl (McNemar), task 3 × 섭동 3 = 9 cell Holm.
- 분모 = 섭동 하 실패했던 에피소드(회복률) + 전체 SR 병기. baseline SR·섭동 SR 은 기측정
  (ppcc: base .72; c1_s200 .375, p1_d003 .44, p2_f040d2 .42 — 48판 기준 성공률).
- β/α 스윕은 fit split 에서만, 확정 후 locked 에서 1회 (3분할 계약 유지 — split_contract.json).
- **fit-eval 에피소드 분리**: steering 방향 fit 은 fit split(짝수 ep), ΔSR 평가는 locked
  (홀수 ep) 섭동 에피소드에서. in-sample rescue 금지 (memory multilayer 사고).

## 3. 데이터 자산 (로컬 = dongkyu PC `/home/dongkyu/pkt_ws/temporal_vla/outputs/eval/robocasa/groot_n15/exp42_induced/`)

| 자산 | 위치 | 내용 |
|---|---|---|
| clean baseline 캡처 | `p0/baseline_cap/` | ppcc_bread 60판(48 succ), action_token_mean 7층(0,2,4,8,10,12,15)+VL(vlln_mean) |
| 섭동 캡처 | `p0/capture/{c1_s200,g1_x015,p1_d003,p2_f040d2}/` | 각 48판, 같은 캡처 kind, perturb_* 메타 포함 |
| 섭동 no-features grid | `p0/grid/` | 15 config 실패율 표 = `p0/p0_failure_rates.tsv` |
| Track I(b4 noise) | `p0/trackI/` | 참고용 (steering 본선엔 불사용) |
| manifests/3분할 | `p0/manifests/` | cal/fit/locked + record_start(시간분리) + split_contract.json |
| fit 산출(conceptor) | `p0/fits_p1/<cfg>/global/...` | perturbed-fail vs succ 대조 (구실험 — exp5-2 는 clean vs perturbed 로 재fit 권장) |
| 프로브 결과 | `p0/clean_vs_perturbed_probe.json`, `p0/layer_geometry_probe.json` | §1 표의 원자료 |
| exp4-1 전달분 | `deliver_expB/` | (자연실패 rescue 용 — 본선 무관) |
| 자연실패 참조 | `natural_strict/` (로컬 사본), 원본 승준 HDD `~/datasets/temporal_vla_outputs/.../phase_event_strict/` | 부차 |
| B1/B3 donor | `b1_donors/`, `b3_donors/` | Track I 용 (본선 무관) |

코드: 브랜치 **`exp/exp4-2-induced-failures`** (worktree `.claude/worktrees/exp4-2-induced-failures`,
dev 미병합) — perturbation.py(4모드+sham), collect `--perturb-spec`·`--instruction-override`,
러너 3종, fit `--ep-subsample-k`·`--record-start-manifest`, 프로브 3종, smoke S1–S5.
**exp5-2 시작 시 이 브랜치를 계승하거나 dev 병합 후 분기할 것.**

## 4. 추가로 모아야 할 데이터

ppcc_bread (로컬 기보유 — 즉시 steering 실험 가능):
1. (선택) **c1_s100 캡처 48판** — 사용자 의견대로 카메라 세기 축소(scale 1.0, 실패율 .36).
   s200 의 d′ 여유(1.1–1.8)로 신호 유지 전망. grid 사이드카는 있음, 캡처만 없음.

drawer left / mixer (**전부 신규** — ppcc 와 동일 파이프라인, cell 파라미터만 §0 표):
2. **scene feasibility**: mixer 는 `analyze/mixer_scene_feasibility.py` 로 seed 스캔 **필수**
   (기하 불가 seed 존재 실측 — 100010 BLOCKED). drawer 는 4-파라미터 이식 후 스캔.
3. **clean baseline 캡처** N=60 (`PHASE=baseline CAPTURE=1`) — 앵커(grasp record)·성공 ep
   집합·clean 짝을 한 번에 확보.
4. **섭동 캘리브레이션 grid** (no-features): C1 scale {0.5,1,2} / P1 δ {3,8,15}cm / P2
   {5,15,40}N×{2,5}rec — **task 마다 실패율이 다르므로 40–70% 게이트 재캘리브레이션 필수**.
   주의: P1/P2 의 trigger 앵커 event 는 task 별로 다름 (drawer 는 grasp:handle 계열,
   mixer 는 27번 문서 라벨러 — `build_perturb_grid.py` 의 anchor key 를 task 별 확장 필요).
   P2-drawer 는 대상 물체가 없어 **P5(drawer damping) 대체 검토** (perturbation.py 에 P5 는
   미구현 — 24b §1.1 원표 참조, DynamicsModder 로 ~30 LOC).
5. **채택 config 캡처** 48판/config + (P1/P2 trigger 스펙 그대로) — capture ON 직수집.

## 5. 실행 환경 주의 (원격 관련)

- **수집·SR eval 은 robocasa Docker 가 있는 로컬(dongkyu PC) 전용** — 기존 원격(승준
  166.104.146.37)은 분석·fit 만 가능 (memory remote-compute-workflow). exp5-2 를 "dongkyu
  원격"에서 돌리려면 그 머신에 robocasa/lerobot 컨테이너·ckpt(/cache)·kitchen assets 셋업이
  선행돼야 함 — **셋업 여부를 첫 단계에서 확인하고, 없으면 수집은 로컬에서·분석만 원격**.
- **결정성은 머신-로컬** — clean↔perturbed 짝은 반드시 같은 머신에서 수집 (섞으면
  state-matched 깨짐. baseline_cap 은 dongkyu PC 산).
- 자원 규칙: 로컬 exp4-1+exp5-2 합산 **GPU 3·serve 6 상한**, GPU당 serve 2. srv48/50 은
  GPU 1개·serve 6 + **산출물 즉시 승준 HDD**. GPU 사용 전 nvidia-smi 로 합산 확인.
- 서브에이전트 모델은 **opus 이하** (fable 금지).

## 6. 권장 실행 순서 (exp5-2)

```
① 환경 확인(원격 셋업 여부·GPU 합산) → 브랜치 계승
② [ppcc, 기보유 데이터] setM fit(clean−perturbed, k=20 서브샘플, fit split)
   + setM_pl + (C1) VL conceptor → 비퇴화·dose 점검
③ smoke: exp4-1 affine hook 로드·항등(β=0)·latch 창 검증 (S5 변형)
④ ppcc 교란-하 ΔSR 파일럿 (c1_s200·p1_d003·p2_f040d2 × L8/L10/L12 사다리, locked ep)
   → 효과 있으면 layer/β 확정
⑤ drawer·mixer 확장: feasibility → clean 캡처 → 캘리브레이션 → 캡처 → fit → ΔSR
⑥ 보고 (confound-audit 규격: in-sample 금지·위약 필수·9-cell Holm)
```
