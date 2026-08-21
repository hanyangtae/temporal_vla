# Handoff — exp6 종결·resample 재시도 정책 축 (2026-08-21)

세션: exp6 online-gated 파이프 (worktree `.claude/worktrees/grid-phase-sep`,
branch `feat/online-gated-pipe`, dev 병합됨 c394c86 이후 커밋 다수 push 완료).
정본 문서: `docs/steering/44_cond_guidance_operator.md` (§7 condg null, §7.1 절제,
§7.2 step-0 β, §7.3 기전, §8 early/resample) + `docs/steering/42` (setM·수축·파이프).

## 1. 확정된 판정 (재론 금지)

- **선형 활성화 개입 전면 null**: setM·수축3종(sconceptor/varc/conceptor)·condg ×
  {β0.3–1.0, last/first denoise call, layer 재선정, 조기발화(절제 detector), 발화−3
  선제(oracle_early)} — 3 task 구제 0/170+, 처치=위약 셀 단위 동일 반복. 기전(§7.3):
  개입은 행동을 바꾸지만(12판 중 9판 발산) 내용이 위약-동질 궤적 교란.
- **resample(발화시점 denoise seed 재추첨 1회, offset 900000) = 캠페인 최초 결정적 구제**:
  4-task 구제 5/65(7.7%) — drawer-L 1/17(s3n1), drawer-R 1/13(s0n0), Dish 1/24(s0n5),
  candle 2/11(s4n0·s9n5). 파손 7/95은 **전부 오발화(FP) 성공셀 재추첨** 매개.
  1회 추첨 순효과 ≈ 0. → **프레임 전환: "감지-후 재시도 정책"(K-재샘플+FP억제)**.
- 시나리오 채점(45 스펙 기준): ① 비파손 조건부 미달(FP-매개) ② 7.7% 미달(K=1)
  ③ 상시-재샘플 대조 미실행(유보) ④ 해당 없음.
- detector: SAFE 신호는 컨택-후 한정(α0.3 붕괴), α0.2=공짜 5record 조기, 절제
  detector(preW 1.00)=조기성 해법이나 rescue 무관. drawer-R unseen-noise FPR 0.64.

## 2. 다음 실험 3종 (사용자 논의 중 — 아직 미지시)

1. **K=3 다중 재시도**: RESEED_OFFSET을 {900000, 700000, 500000} 세 arm으로 돌려
   셀별 union 구제 = 1−(1−p)^K 근사 (p≈0.08 → K=3 ~22%, 기준 ② 사정권).
   러너 그대로: `ARMS=resample RESEED_OFFSET=<x> OUT_ROOT=<별도>`.
2. **상시-재샘플 대조**: `--reseed-from-record 0` (TRIGGER_TSV 무시하고 K=0부터) —
   기준 ③(타이밍 정보 우위)·detector 사슬 존재 이유 판정. 러너에 arm 추가 필요(소).
3. **FP 억제 sweep**: 성공셀 보호 — FAILURE_ALPHA 0.05 또는 절제 detector 조합으로
   trigger 표 재생성 후 resample 재실행.

이월 진단(#8): **margin 자가평가 캡처 2판** — `launch_margin_diag.sh`(jobs tmp에 있음,
CAPTURE_FEATURES=1 + 좌표 인자 배선 완료·커밋됨) → pkl에서 L15 d0 margin 재계산,
개입 후 하락 여부 = write 실패 층위 판별. 분석 후 pkl 삭제.

## 3. 인프라 좌표 (그대로 쓰면 됨)

- 러너: `scripts/steer/online_gated/run_online_gated_eval.sh` — STEER_OP=
  setpoint_seg|conceptor|condg, arm oracle_early/resample(TRIGGER_TSV 셀별 타이밍,
  EARLY_OFFSET/RESEED_OFFSET), CAPTURE_FEATURES=1(진단 캡처+좌표), CONDG_LAYER/
  CONDG_APPLY_CALL. EP_MODE=replay가 표준(수집 머신 매칭 필수).
- trigger 표: kanu `outputs/eval/robocasa/groot_n15/og_condg_d0_b10_merged/logs/
  triggers_OpenDrawer_left.tsv`(절제 det), srv48/50 `~/pkt_ws/temporal_vla/outputs/tmp/
  triggers_{DishwasherRack_out,OpenDrawer_right,PPCC_candle}.tsv`(full det).
- 머신 매칭: drawer-L=kanu, Dish=srv48(GPU0), drawer-R·candle=srv50(GPU2).
  A100 serve ~6GB(8/GPU 가능하나 **동시부팅 실질 한계 6** — 240s 시차 부팅).
  kanu는 빈 GPU만(compute-apps PID로 타인 확인, <1GB 잔여도 사용 중일 수 있음).
- 집계: `final_agg_condg.py`(jobs tmp — 레포 이관 안 됨, 셀-paired 구제/파손+in-fit/
  held-out-noise 2분할). scene 분할 병합은 cells tsv cat + raw_rollouts cp -rl.
- 갤러리: `outputs/eval/robocasa/groot_n15/video_gallery/` (http://166.104.35.33:8898).
  annotate는 h264 자동. **caption burn-in 절대 금지** — dev 차단판 유지, 장시간 스크립트는
  /tmp 금지(청소로 사망), robocasa 컨테이너 NVML 죽으면 docker restart.

## 4. 미결

- OvenRack(데스크탑) 오염 영상: 비결정이라 재생성 불가 — 삭제 vs 유지 사용자 지시 대기.
- 시나리오 스펙(45, feat/kai-paper): v3 수집 축(scene 고정×env_seed 20-30×inf 2-4)
  미수집 — grid_v2는 scene×denoise-seed 확장이라 불일치 확인됨.
- candle step-0 게이트: L15 transport 0.57(소표본) 경계 1건 — 미진행 권고 상태.

## 5. 새 세션 프롬프트 (복붙용)

"worktree .claude/worktrees/grid-phase-sep (branch feat/online-gated-pipe)에서 exp6
후속을 진행해줘. 정본은 docs/collab/handoff_20260821_exp6_resample.md — §1 판정은
재론 금지, §2의 실험 3종(K=3 재시도·상시-재샘플 대조·FP 억제)과 margin 캡처 진단이
후보다. 사용자 지시를 받은 것부터 §3 인프라 좌표대로 발사하고, 보고는 셀-paired
구제/파손 + in-fit/held-out-noise 2분할로."
