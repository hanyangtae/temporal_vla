# Handoff — rs_steer 파일럿 완주·activation 궤적 인프라 (2026-08-25)

세션: 메인 세션(main checkout `~/pkt_ws/temporal_vla`, branch **feat/rs-steer-v4** =
최신 dev 908dc24 + exp6 전체 병합, push 완료 df00ba5). 이전 정본:
`docs/collab/handoff_20260821_exp6_resample.md`(§1 판정 재론 금지 유지) + `docs/steering/44`.
국내투고 세션은 별도 worktree로 이동(main checkout은 이 라인 소유).

## 1. v4 지터 격자 (데이터 기반 — 확정)

- **k(지터) 축 메커니즘**: reset(seed)는 무효(make seed가 전부). 성립 축 =
  **set_ep_meta 주입 + 연속 reset() k(=reset_idx)** — 물건종류·target·instruction 고정,
  **배치만 변주**(로봇 base pos는 ep_meta로 고정, 관절 지터는 mm~cm 미미 — "로봇 자세
  축"은 사실상 없음, 45 스펙 문구 정정 필요). drawer는 left/right만 k마다 재추첨 →
  k-사전스캔으로 해결. (base, ep_meta, k) bit 결정적.
- **수집**: n15_grid_v4 (plan 46ea62d53e09) — 10 task × s0-4 × n0-4 × k5(신규 4 + base 재사용)
  = 1,250판. 정본 index = `outputs/steer/online_pipe/manifests/index_rollouts_v4.tsv`
  (평탄 cell_si = scene*100+k, base행 = +99, jitter_reset_idx 열). pkl은 승준 HDD
  `/home/kimseungjun/datasets/temporal_vla_store/groot/n15/grid/`(v4=46ea…, base=v1/v2 plan 디렉토리).
  ep_meta 50개 = `<grid>/46ea…/ep_meta/<task>/<env--seed>.json` (로컬 사본
  `outputs/collect/ep_meta_v4/` + cell-id symlink 계층).
- **k가 성패 지배**: 같은 scene에서 k만 바꿔도 5/0↔0/5 (drawer-L s4 k4=5/0 vs k6·k9=0/5 등)
  — 실패의 상당수가 배치-결정형.

## 2. 구제 케이스 선정 기준 (사용자 확정) · 파일럿 결과

기준: ① scene succ>5 ② 대상 k succ 1~4 ③ fit = 나머지 k 전부 + 대상 k 실패(in-sample 명시)
④ 나머지 k에도 실패 ≥1 (k-좌표 학습 차단). → **전체 51케이스/118판**
(`outputs/steer/online_pipe/manifests/v4_rescue_cases.tsv`), 파일럿 = instruction당
1케이스 1판 (`v4_pilot_cases.tsv`, 선정기 `scripts/steer/online_gated/select_rescue_cases.py`).

**파일럿 8케이스 판정** (arm: base / rs_early(발화−3 재추첨) / rs_setm β1.0 /
rs_condg step-0 강제등록 β1.0 / rs_coast 대조 conceptor global β0.5; detector = phase-gt
절제 통일, coffee만 full det 예외):

| 케이스 | base | 결과 |
|---|---|---|
| oven s4k1 | 재현·발화67(push-in=역행 라벨, out task 정상 어휘) | 전 arm 실패 |
| bread s0k0 | 재현·발화31 | 전 arm 실패 |
| **marshmallow s3kbase** | 재현·발화5 | **rs_condg 구제(succ, r94 조기종료)** — 유일. 나머지 arm 실패 |
| jug s4k0 | 재현·발화22 | 전 arm 실패 |
| candle s3k0 | 재현·**미발화**(full·절제 공통) | 개입 정의 불가 |
| dish s4k1 | **재현 불일치**(replay 2회 연속 succ) | 판정 보류 — 조사 대상 |
| coffee s2k0 | full det 예외 진행 중단 상태 | 미완 |
| drawer-L s4k2 | **replay instruction 반전**(left→right) | 제외 — reset off-by-one 의심, 조사 대상 |

핵심 소견: ① 유효 5케이스 중 구제 1(조기발화 재추첨+condg) ② **read≠write 시각 사례**:
oven rs_setm 개입 후 사후 검출 p 0.010≪δ0.177(실패 신호 소거)인데 실제는 실패.

## 3. activation 궤적 영상 인프라 (완성, 커밋 df00ba5)

- **아티팩트**: https://claude.ai/code/artifact/4fbe7cf9-4c77-4230-bcd4-39e97ec03f1e —
  17편(4케이스×arm), 960×780: fit-pool PCA 배경(초록succ/빨강fail) + 궤적(주황=개입) +
  record/phase(GT labeler=개입이 참조한 값)/활성 연산자/검출 p·δ + 로봇 3-view 세로 컬럼.
  페이지 생성기 = jobs tmp `build_traj_page.py`(세션 종속 — 필요시 영상 dir에서 재작성).
- **렌더러**: `scripts/analysis/grid_phase/render_activation_traj.py` — 검출값: base=기록,
  arm=**사후 재계산**(base 4판 bit-일치+trigger 4/4 재현 검증 게이트 통과 후 적용,
  `--detector-task`). 실행은 **lerobot 컨테이너**(matplotlib+torch+ffmpeg; robocasa엔
  matplotlib 없음), 한글폰트 docker cp NanumGothic.
- **배경 PCA**: `outputs/analysis/v4_pilot_viz/*.npz` (케이스×{setm(L12/last), condg(step0/L)}),
  승준 산출. 영상 `outputs/analysis/v4_pilot_viz/videos/`.
- **진단 pkl** ~12GB: `outputs/eval/robocasa/groot_n15/og_v4_pilot_cap{,_jug}/` — 규약상
  분석 후 삭제 대상, **사용자 확인 대기**.

## 4. 인프라 좌표·함정 (신규 확립분)

- **fit**: 케이스별 매니페스트 fit — `select_rescue_cases.py --fit-manifest-dir`(경로 기반,
  plan_id 혼재 시 (scene,noise) 튜플 선택은 **조용한 오귀속**이라 금지) →
  `fit_cond_guidance.py --episode-manifest --force-register --min-eps-per-class N`(극소표본
  딱지) / setM=fit_setm L12 seg / coast=`fit_contraction_ops --ops conceptor --phase-groups
  global`(6/8 퇴화 — COAST global은 이 데이터에서 활성화 소거기; candle·marshmallow만 생존).
  NPZ 트리 = `outputs/steer/online_pipe_v4_pilot/<slug>(심링크→케이스dir)/{case,case_setm}`.
- **eval**: 러너 v4 replay(`--jitter-reset-idx`+`EP_META_DIR`+`EP_META_LOAD_ENV_NAME`,
  ep_meta는 `<root>/<task>/<slug>/` symlink 계층 필요), trigger 표 = base arm(phase-gt det)
  → `make_triggers.py`(**cell_key 열 우선** — scene_idx는 base 접힘), rs arm 공통 st=trig−3.
  재개 스크립트: kanu `outputs/tmp/og_v4_resume_kanu.sh <GPU1> [GPU2]` / srv
  `~/pkt_ws/temporal_vla/outputs/tmp/og_v4_resume_srv.sh <GPU>`(hostname 자동 케이스,
  멱등, SERVE_BOOT_TRIES=360 권장).
- **진단 캡처**: `CAPTURE_FEATURES=1 CAP_GRID=0 DETECTOR_LAYERS=12,15` →
  `--diag-unplanned`(plan 검증 생략, diag/ 하위, export-horizon 완화). **detector 없는
  serve도 캡처 플래그 강제**(92c9f9f — 아니면 pre-decode 1024d가 나와 fit 공간 불일치).
- **함정**: 러너 파일 실행 중 수정 금지(열린 bash 오프셋 깨짐 — 재확인) · 호스트 pkill은
  컨테이너 내 python 못 죽임(docker exec pkill 필요) · ssh 원격 setsid는 `ssh -f`로 ·
  worker_w0.log는 호출마다 truncate(에러는 단독 실행으로 잡을 것) · diag 모드 rc=13은
  집계 오탐(판정=pkl 존재) · A100 타인 부하 시 serve 로딩 12분+(SERVE_BOOT_TRIES).

## 5. 다음 후보 (미지시 — 사용자 결정 대기)

1. **51케이스 본 라운드**: 파일럿 배선 그대로 확장(케이스별 fit 51 + eval ~118판×arm).
   marshmallow rs_condg 구제 1판의 재현/일반화 검증이 1차 질문.
2. **dish 재현 불일치·drawer-L instruction 반전 조사** (replay 신뢰성 이슈 2건).
3. 진단 pkl 12GB 삭제 여부.
4. 보류분: K=3 다중 재시도·상시-재샘플 대조·FP 억제(#11-13)·margin 자가평가 진단(#8).

## 6. 새 세션 프롬프트 (복붙용)

"main checkout ~/pkt_ws/temporal_vla (branch feat/rs-steer-v4)에서 rs_steer 라운드 후속을
진행해줘. 정본은 docs/collab/handoff_20260825_rs_steer_pilot.md — §2 파일럿 판정과 §4
인프라 좌표를 그대로 쓰고, §5 후보 중 사용자 지시를 받은 것부터. 이전 라운드 확정 판정
(handoff_20260821 §1: 선형 개입 null·resample 5/65)은 재론 금지."
