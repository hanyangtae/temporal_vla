# 핸드오프 — grid v6: scene(주방)·jitter(j) 층위 재정의와 재수집 (2026-09-03)

정본: 이 문서(격자·수집 방식·폴더 구조) + `docs/04_data_storage_convention.md` §3.1.1(저장 규약).
이전 라운드(v5, plan e6b316053d1c 계열, 5주방·평탄 seed scene·k 지터)는 **legacy** 로 아카이브에 남긴다
(재현 시 옛 5주방 env 설정 필요 — §6). v6 부터는 아래 계약만 쓴다.

## 1. 층위 (사용자 확정 2026-09-03)

| 층 | 정의 | 무엇이 바뀌나 |
|---|---|---|
| instruction 키 | 과제 + 문장 변형 + (pull 계열) 스폰 side | 12키: oven-left/right, washer-left/right, drawer-left/right, ppcc apple/jug/candle/bread/marshmallow, coffee |
| **scene s** | **주방 = (layout, style) 쌍** (+ pull 계열은 스폰 side 로 갈린 키에 속함) | 주방 도면·외관·fixture 배치. style 만 따로 바꾸지 않는다(target split 은 layout↔style 고정) |
| **jitter j** | 같은 scene 의 세계 변형 하나 | PPCC·coffee: ep_meta 고정 + 연속 reset(물체 위치·팔 관절). pull 계열: **base 오프셋**(+drawer 는 reset 재추첨도) |
| noise n | 정책 denoise seed | 세계 불변 |

- side(오븐·식기세척기): 문을 연 채 시작해 정면 앵커가 막히고 로봇이 좌/우 약 0.45m 로 밀려난 결과.
  seed 에 묶인 50/50 무작위(rack 단·in/out 과 독립). **로봇 시점** 좌/우로 판정한다:
  `l = (−sin yaw, cos yaw)`, `lat = l·(base − fixture_pos)`, lat>0 → left, else right.
  서랍의 left/right 는 스폰이 아니라 **대상 서랍**(문장 변형)이다. PPCC·coffee·drawer 는 열린 문이
  없어 스폰이 단봉(±15cm)이라 side 축이 없다.
- 규모: 12키 × scene 3 × j 5 × n 5 = **900판** (~540GB).

## 2. 주방 집합과 scene 선택

- env 주방 목록 = **target split 10주방** `layout_and_style_ids = [[1,1],…,[10,10]]` — gym 래퍼
  `create_env(layout_and_style_ids=…)` 인자(또는 `ROBOCASA_LAYOUT_STYLE_IDS`)로 지정, plan
  `extra.env_kwargs` 에 기록. **목록이 바뀌면 seed→주방 추첨이 바뀐다** — legacy 5주방 목록
  `[[1,1],[2,2],[4,4],[6,9],[7,10]]` 은 v5 이하 전용.
- task 별 가능 layout(스캔 `outputs/analysis/seed_scan/fixture_groups/*_target10_*.tsv`):
  오븐 = {2,4,7,9}("oven rack out" 은 L4 뿐, L2·7·9 는 bottom/top 문장 → **bottom 문장 인정**),
  식기세척기·서랍·coffee = 1~10 전부, PPCC = 물체당 layout 1~2 seed(추가 스캔 100600~102999 진행).
- 공통 축: **L4·L9** + 오븐은 L7, 나머지는 L5 (PPCC 는 스캔 결과로 확정). scene 은 키당 3.
- scene 의 seed: 스캔 TSV 에서 (layout, 문장, side) 조건을 만족하고 feasibility(랙/서랍 스윕)를
  통과한 seed 1개. 같은 scene 의 j 는 그 seed 의 ep_meta 를 공유한다.

## 3. jitter j 정의 (재현 계약)

| 키 | reset_idx(연속 reset 횟수) | base 오프셋 (lat = 로봇 좌우, 지정 방향; back = 뒤로) |
|---|---|---|
| ppcc 5종, coffee | j (0..4) | 없음 |
| drawer-left/right | scene 별 채택 reset 목록[j] (연속 reset 이 서랍 좌/우를 재추첨하므로 문장이 맞는 인덱스만 — 선택표 `reset_idx_list`, v5 k-스캔과 동일 원리) | back ∈ {0, 0.05, 0.10, 0.05, 0.10}[j], lat 0 |
| oven-·washer-left/right | 0 | (lat, back) ∈ {(0,0), (0,0.05), (0,0.10), (0.05,0.10), (0.05,0.15)}[j]; **lat 은 항상 fixture 중심 쪽**(left 키 → 오른쪽으로, right 키 → 왼쪽으로). 안쪽 lat 만(back 0)은 열린 문과 접촉해 불가(전수 reset 검사 실측) |

- 적용: `reset(seed)` → ep_meta 획득 → `init_robot_base_pos += (−f·back) + (±l·lat)` (f = (cos yaw, sin yaw),
  l = (−sin yaw, cos yaw), 부호 = side 규칙) → ep_meta 주입 → plain reset (reset_idx+1)회 →
  **충돌 검사**(RoboCasa 원 스폰의 접촉 상태 대비 **새 접촉 또는 1cm 이상 관입 증가**면 RuntimeError — 원 스폰 자체가 열린 문에 닿아 있는 scene 이 있어 '접촉 有' 기준은 못 쓴다). 오프셋값·최종 base 를
  meta.json 과 셀 ep_meta 에 기록.
- 재현: eval/replay 는 plan 의 같은 (scene, j) 정의에서 오프셋을 **다시 계산**해 같은 절차를 밟는다
  (JSON ep_meta 사전 주입 금지 — v5 게이트 실측). 기록된 base 와 재계산 base 가 다르면 fail-loud.
- pull 키의 j 는 reset 재추첨을 쓰지 않으므로 팔 관절도 고정이다(변화 = base 만).

## 4. 폴더 구조 · plan · 인덱스

```
<plan_id>/<machine>/<instruction 키>/s<sid>/j<jid>/n<nid>/<arm>/{rollout.pkl, traj.csv, video.mp4, meta.json}
<plan_id>/ep_meta/<task>/<env>--seed<es>.json
```
- sid = plan `scenes[key]` 순서, jid = `jitters` 순서(**reset_idx 가 아니라 j 인덱스**), nid = noise 순서.
- plan (CollectionPlan 확장, legacy 필드 유지): `instructions[key]`(scene 별 env_seed, 순서 = sid),
  `noise_seeds`, `scenes[key][sid] = {layout, style, side, lang, fixture_group, spawn_lat}`,
  `jitters[key][sid][jid] = {reset_idx, lat, back}`, `extra.env_kwargs.layout_and_style_ids`,
  `extra.machine_assignment`. 셀 키 `key|s<sid>|j<jid>|n<nid>`.
- 인덱스 열: scene_idx, jitter_idx, noise_idx + 출처 열(env_seed, inference_seed, jitter_reset_idx,
  base_lat, base_back, layout_id, style_id, side, lang). 구 `jitter_reset_idx` 3축(k 층)은 legacy 읽기만.
- collector 좌표 인자: `--scene-idx --jitter-idx --noise-idx --grid-instruction` (+plan) — reset_idx·오프셋·
  문장은 plan 에서 읽는다(`resolve_grid`). 문장 대조는 scene 의 `lang`.

## 5. 머신 배정 (= replay 홈)

| 머신 | 키 | 판수 |
|---|---|---|
| srv48 (1장 × serve 6) | oven-left, oven-right, washer-left, washer-right | 300 |
| kanu (GPU ≤3장 × serve 2) | drawer-left, drawer-right, coffee | 225 |
| srv50 (1장 × serve 6) | ppcc apple·jug·candle·bread·marshmallow | 375 |

발사 전 `docs/05_gpu_server_rules.md`(lease) + 첫 셀 게이트(A 수집 / B 재실행 / D eval 경로) 통과.
pull 키는 게이트에 **base 재계산 일치** 항목을 추가한다.

## 6. legacy (v5 이하) 취급
- 아카이브 `e6b316053d1c`(v5 k-층)·구 plan 들은 그대로 두되 인덱스에 `legacy=1`. 재현하려면 래퍼 기본
  5주방 목록으로 돌려야 한다(v6 plan 의 env_kwargs 와 다름). 분석 정본은 v6.
