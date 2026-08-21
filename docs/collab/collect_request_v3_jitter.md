# 수집 요청 — n15_grid_v3: 배치·로봇자세 지터 축 (2026-08-21, exp6 세션 → 데이터 추가 수집 세션)

요청 세션: exp6 online-gated (worktree grid-phase-sep, branch feat/online-gated-pipe).
근거 스펙: `docs/steering/45_hotfix_scenario_spec.md` §1(재발 변주 3축)·§4(v3 제안).

## 1. 목적 (왜 이 축인가)

구제 시나리오의 재발 변주는 ① 정책 확률성(denoise seed) ② **물건 배치 지터** ③ **로봇
초기 자세 지터**의 3축인데, 현 grid v1/v2는 셀당 env_seed 1개 고정이라 ①만 커버한다
(45 §4 주의: v2 plan은 v1의 scene×denoise 확장이라 이 축이 아님 — 별도 plan 필요 확인됨).
v3 = **같은 (task, scene, instruction)에서 env_seed만 변주**해 ②③을 확보. 이 데이터로
per-(task×scene) 연산자 fit(scene당 표본 확보)과 episode-축 held-out eval을 한다.

## 2. 선행 검증 2건 (수집 전, 필수)

1. **reset-dump 검증 (10분)**: 대상 scene의 ep_meta 고정 + env_seed 변주로 reset 여러 번
   덤프 → **물건 종류·주방 구조 불변, 배치(reset_region 내 위치)와 로봇 초기 관절만 변화**
   확인. 근거: `src/benchmarks/robocasa` kitchen.py L818 — ep_meta는 reset_region(규칙)만
   저장, 실제 배치는 reset마다 placement_initializer.sample() 재추첨 + robosuite
   initialization_noise="default".
2. **env_seed 사전 스캔**: seed마다 task variant가 바뀔 수 있음(OpenDrawer 좌/우, PPCC 물체
   등 — 기존 실측). 정책 없이 reset만으로 **instruction·variant 불변인 seed만** 채택.
   ep_meta 고정 하에서도 variant가 흔들리는지가 검증 1의 확인 항목에 포함.

검증 1에서 "ep_meta 고정+env_seed 변주로는 배치가 안 바뀐다"로 나오면 즉시 중단하고
회신 요청 (프로토콜 재설계 필요).

## 3. 수집 사양

| 축 | 값 |
|---|---|
| task (2종) | OpenDrawer/left, PPCC/candle |
| scene (task당 3개, 기존 grid 실측 SR 중간대) | drawer-L: v1 scene0(es 100001, SR0.40)·scene8(es 100017, 0.50)·scene9(es 100019, 0.60) / candle: scene4(es 100214, 0.40)·scene9(es 100741, 0.30)·scene2(es 100154, 0.70) |
| env_seed (신규 축) | scene당 20개 (사전 스캔 통과분; 위 base env_seed의 ep_meta 고정) |
| inference_seed | 2개 (기존 규약 대역에서: 예 n0=200000, n1=300000 계열) |
| 판수 | scene당 40 → task당 120, **총 240판** |

- 추론 표준: GR00T N1.5, chunk 16 예측/5 실행(기존 grid 수집과 동일 serve 설정).
- **수집 머신 = 향후 replay eval 머신 매칭 필수**: OpenDrawer/left=**kanu**,
  PPCC/candle=**srv50** (데스크탑 금지 — 단일GPU 렌더 비결정).
  kanu 발사 규칙(타인 GPU 금지·GPU당 serve≤2·≤3GPU), A100 규칙(GPU당 serve 6, 동시부팅
  6·240s 시차) 준수.
- **overlay_text OFF 차단판 확인** (dev 4d8c9ad 이후 강제 OFF — 병합돼 있는지 확인).

## 4. 산출물 (회수 대상)

1. **index_rollouts_v3.tsv** — 기존 index 스키마 그대로 + env_seed 열이 셀 내 변주임이
   드러나게 (scene_idx는 base scene 기준, env_seed 열이 실제 reset seed). replay 재생이
   env_seed·inference_seed로 결정되는 기존 계약 유지 (러너 변경 불요 확인됨).
2. **pkl 캡처 포함** (연산자 fit용): docs/04 규약 — sig 내용지문, 절대경로 기록 금지,
   캡처밀도 5열, 기존 grid와 동일 레이어/토큰 규격 (DiT residual, all_token_full).
3. 셀별 succ/fail 판정 tsv (판정 사이드카 포함).
4. 검증 2건의 결과 요약 (reset-dump 확인 내용, 채택/기각 seed 목록).

## 5. 회신·후속

- 완료 시 이 문서에 결과 경로 추기하거나 exp6 세션에 전달. 이후 fit(per-scene 연산자)·
  eval(episode-축 held-out, 재샘플+연산자 결합 arm)은 exp6 세션이 수행.
- 질문/차단 사항은 이 파일에 코멘트 추가 후 사용자 경유로 전달.
