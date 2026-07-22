# wrist cam attention — step/phase별 카메라 뷰 attention 기여도 (관측)

작성 2026-07-16. 곁가지 관측 실험. 단일 출처 코드: `scripts/serve/attn_hooks.py`,
`scripts/safe/groot_n15/robocasa/{collect/collect_cam_attn.sh,analyze/cam_attn_phase.py,vis/cam_attn_timeseries.py}`.

## 질문

"wrist cam은 trajectory 내내 유의미한 정보를 주진 않을 것 같은데, step/phase별로 각
카메라 image의 attention 비중이 어떻게 달라지는가?"

## 측정 방법

GR00T N1.5 RoboCasa. 카메라 3뷰(left/right = agentview, wrist = eye-in-hand),
Eagle에서 뷰당 256 vision token. **DiT cross-attention**(action head가 VL 시퀀스에
attend하는 8개 짝수 block)의 attention weight를 뷰별 token 블록으로 그룹 합산 →
"action 생성이 어느 카메라 위치를 읽는가"를 mass fraction으로 산출.

- 뷰 경계는 `eagle_input_ids`의 image token 위치(뷰당 256개, wrist=마지막)로 매 요청 복원.
- query축은 [state(1) | future(32) | action(16)]로 그룹화, 아래 지표는 **action query** 기준.
- 시간 해상도 = 5 env-step(chunk 5 실행), phase는 6/7-phase event labeler GT.
- 지표 = **wrist share of vision** = wrist mass / (left+right+wrist). uniform 기대 = 1/3
  (뷰당 token 동수). vision-share로 정규화해 text token 수 가변을 배제.
- 수집: `ppcc_bread`(PnP, 20 ep) + `pq3_drawer_right`(OpenDrawer, 20 ep), fixed-scene,
  inference noise 변주. GPU 2. capture ON/OFF에서 action bit-identical 검증 통과.

## 결과

**"내내 무의미"는 반증됐다.** wrist는 세 뷰 중 항상 attention 최대이고, vision-share가
전 구간 uniform(0.333)을 크게 웃돈다 (0.40~0.50). 즉 모델은 rollout 내내 wrist를
상당히 참조한다.

**step/phase별로 변한다** (bread 기준, action query, ep-mean±bootstrap CI):

| phase | wrist share (succ) |
|---|---|
| reach-to-object | 0.440 |
| grasp | 0.460 |
| wrong-grasp | 0.472 (fail만) |
| insert-settle | 0.452 |
| place | 0.406 |

- **grasp 계열에서 최고**(0.46~0.47), **place(놓기)에서 최저**(0.41). 손이 물체를 잡는
  국면에 wrist 의존이 가장 크고, 팔을 뻗어 목표 위치에 놓는 국면에 외부뷰 쪽으로 이동.
- 시계열에서도 reach→grasp 상승, transport 초반 하강, place 저점, insert-settle 재상승의
  출렁임이 에피소드마다 재현 (`vis/*--ep*.png`, `--overlay_wrist_share.png`).
- drawer는 더 평평(0.40~0.44) — 서랍 손잡이 조작이라 phase별 변동이 PnP보다 작음.

**depth 의존** (`vis/wrist_share_by_block.png`): 얕은 DiT block에서 wrist 편중이 약간 더
큼(block0 0.46 → block14 0.42), 완만한 단조 감소.

**succ vs fail**: 대부분 phase에서 CI 겹침 — wrist attention 자체는 성패를 가르는
신호가 아님. (fail-only인 wrong-grasp가 0.47로 높지만 phase 정의상 fail에만 존재.)

## caveat (해석 한계)

1. **위치 귀인 ≠ 순수 카메라 정보**: DiT 앞 `vl_self_attention`이 VL 시퀀스를 한 번 섞어,
   각 위치에 다른 뷰 정보가 일부 혼입될 수 있다. mass는 "그 카메라 위치를 참조"이지
   "그 카메라 정보만 사용"이 아니다.
2. **관측이지 인과가 아님**: attention이 높다고 그 입력이 행동을 좌우한다는 보장은 없다
   (wrist를 gray/frozen으로 치환하는 인과 probe는 이번 범위 밖 — 필요 시 후속).
3. text 그룹이 항상 최대(~0.30)지만 본 실험 질문(카메라 간 비교)에서는 vision-share로 분리.

## 산출물

- `outputs/eval/robocasa/groot_n15/cam_attn/analysis/cam_attn_records.csv` (3255 record)
- `.../analysis/cam_attn_phase_agg.json` (cell×phase×succ 집계)
- `.../vis/*.png` (per-episode 시계열, overlay, phase bar, block-depth, smoke heatmap)
