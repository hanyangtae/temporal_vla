# Handoff — per-step 게이팅·cluster-k8·v4 정렬 라운드 (2026-08-31)

세션: 메인(main checkout `~/pkt_ws/temporal_vla`, branch **feat/rs-steer-v4**, 최신 push
c409d9d+). 정본: [`docs/steering/47_perstep_gating_pipeline.md`](../steering/47_perstep_gating_pipeline.md)
(게이팅 설계) + 이 문서(라운드 실측). latch 계열 문서·판정은 폐기됨(47 머리 참조) — 재론 금지.

## 1. 확정 판정 (이 라운드 실측)

1. **per-step 게이팅 구현 완료** (47 §3): serve가 매 record 1차 무개입 pass로 y_t 판정
   → 발화 시 **DiT-only 2차 pass**(backbone 캐시, `types.MethodType` wrap)로 action 교체
   → detector 롤백 후 x_t′ 재step(y_t′ 기록·h′ 커밋). 스모크 5/5(배관 동치 max|Δaction|=0,
   발화 재현, 개입 인과, y′, 2회 bit 재현). y−y′ = read-erasure 지표(setM만 ≫0).
2. **phase 정의 = task별 K=8 activation cluster** (ae_cluster: mean-center+scalar std →
   AE 1536→256→256→16 → per-task KMeans k8; GT event labeler는 성공판정·gt_phases 병행
   기록용으로만). serve가 활성화에서 cluster 자체 판정(`--cluster-phase-bundle`).
3. **★v4 수집 성패 라벨은 경계 판에서 신뢰 불가**: 수집-실패 판의 ~55%가 base replay에서
   성공(수집=고부하 병렬 비결정 추정). 반면 **replay 자체는 완전 결정적** — base 2회
   36판 불일치 0, v2판↔v4판 base 교차 65/65 일치. → **판정 프레임 = base-replay 재정박
   paired**(구제=base실패→성공, 파손=base성공→실패)로 확정.
4. **★v4 정렬(detector·cluster를 eval과 같은 v4 분포로 학습)이 승부수**:
   | arm | v4판 구제/파손 (68 pair) | v2판 (65 pair) |
   |---|---|---|
   | ps_reseed | **5/29 · 5/37** | 2/28 · 10/34 |
   | ps_setm | **6/29(21%) · 9/34** | 3/29 · 12/33 |
   | ps_condg | 1/26 · 4/20 | 3/27 · 6/22 |
   구제 ~2배·파손 절반. **setM 최초 구제**(전 라운드 통틀어; bread2·marsh2·candle2),
   reseed 순효과 균형 도달. 파손의 주범 = detector 동작점 불일치(FP)로 확정.
5. detector 교훈: v1→v2 재학습만으로 task별 동작점 요동(bread FPR 0.06→0.67 = 실전
   발화 1→117회). v4판 ckpt는 **α 0.05/0.1/0.2 3종 저장**(FP 레버). cluster 절제는
   GT 대비 무해·소폭 열세(pooled TPR 0.93 vs 0.99), 길이 편승 없음(D3·시뮬 양쪽 확인).
6. 구조 제외: **apple**(v4 실패 판 0 — detector 정의 불가), **coffee**(성공 9판 중 8판
   s2 편중 — scene-disjoint CP 밴드 증명적 불가), **drawer-L**(replay instruction
   좌우 반전 미해결 — 클라이언트 fail-loud로 대부분 결손).
7. GT-phase per-step 파일럿(중간 기록): marshmallow reseed 구제 + 폐루프 자연 침묵
   실증, oven rs_setm read-erasure 사례. 영상 아티팩트
   https://claude.ai/code/artifact/4fbe7cf9-4c77-4230-bcd4-39e97ec03f1e (GT 파일럿 기준 —
   cluster/v4판으로 미갱신).

## 2. 인프라 좌표 (v2·v4 두 벌, 접미로 구분 — 덮어쓰기 금지 규약)

- **번들**: `outputs/analysis/grid_phase/ae_v4_k8/ae_bundle_v4_k8.npz` (v2판: `ae_v2_k8/ae_bundle_k8.npz`).
  판정기 = `src/failure_online/cluster_phase.py` (serve `--cluster-phase-bundle/-task`);
  fit 어댑터 = fit_setm/fit_cond_guidance `--cluster-bundle/--cluster-task`.
- **detector**: `outputs/analysis/grid_phase/detector_v4/{cluster-k8,phase-gt-v4}/`
  (8 task, 분할 3/1/1 seed0 — v4는 scene 5개라 기본 6/2/2는 퇴화; excluded_tasks.tsv 참조).
  v2판: `detector_v2/{cluster-k8,phase-gt-v2,cluster-k8-fix}`.
- **연산자**: `outputs/steer/online_pipe_v4_pilot/<case>/{case_setm_ck8v4,case_ck8v4}`
  (51케이스; v2판 `case_*_ck8`; GT판 `case_setm`/`case`). condg layer = `case_ck8v4/layer_choice.txt`.
- **eval 결과**: `outputs/eval/robocasa/groot_n15/` 아래 `og_ps_pilot`(GT 파일럿),
  `og_ck8_pilot`, `og_ck8_expand{,_srv50}`(v2판), `og_ck8v4_expand{,_srv50}`(v4판, kanu몫+srv몫).
  판 목록 정본 = `outputs/steer/online_pipe/manifests/v4_expand_eval.tsv`(118판, machine 열),
  케이스 정본 = `v4_rescue_cases.tsv`(51), fit 매니페스트 = `v4_fit_all/`.
- **오케스트레이터**(untracked, outputs/tmp/): `og_ck8v4_expand_kanu.sh <GPU...>`(케이스별
  NPZ 라우팅 심링크 `.roots/`, ps_base2 안정성 arm 포함) / srv판 `og_ck8v4_expand_srv.sh`
  (srv 기기 내, hostname으로 worker1/2 자동). 전부 멱등.
- **serve/클라이언트 계약**: payload `perstep_gate{op,reseed_offset}`·`perstep_debug_rerun`;
  응답 `features.{failure_*,failure_score_post,perstep_fired/op/seed2/cluster,gate_skipped}`;
  클라 `--gated-steering-mode perstep --perstep-op {none,reseed,setm,condg}
  --perstep-cluster-phase`; 러너 env `CLUSTER_BUNDLE` 통로.
- **srv 접속**: `AISem_50_junhyeong`(=worker2)·`AISem_48_junhyeong`(=worker1), ssh config
  등록됨. **serve = host conda** `~/miniconda3/envs/lerobot_050_groot/bin/python` +
  `SERVE_PYTHONPATH=~/pkt_ws/temporal_vla/lerobot/src`, A100 serve 6/GPU·빈 GPU만.
  repo = `~junhyeong/pkt_ws/temporal_vla` (연산자/번들/ckpt는 tar로 반입 — git엔 없음).
- **원격(승준) 산출**: segA_v4 = `~/datasets/.../analysis/grid_phase_v4/segA{,_v4_ck8}`
  (~120G, 분석 후 정리 검토 대상), 체인 로그 `~/workspace/temporal_vla/outputs/tmp/v4_align/`.

### 함정 (이번 라운드 신규)
- **docker NVML 상실**: 장기 가동 lerobot 컨테이너가 GPU를 잃음(`Failed to initialize
  NVML`) → serve가 CPU로 떠서 FlashAttention 에러 → `docker restart lerobot`. srv에도 동일.
- **pkill/pgrep self-match**: 명령줄에 대상 문자열이 있으면 자기 자신 kill(로컬·ssh 원격
  모두 당함) — 패턴 변수 분리 또는 kill/발사 ssh 분리.
- `set -u`에서 `local a=$1 b=...$a...` 한 줄 선언은 $a 미정의 — 줄 분리 필수.
- v4 shard 추출은 지터-인식 extract_grid_matrix(c409d9d) 필수(좌표 중복·bread union).
- EVAL_NOISES 다중 배칭은 replay 결정성에 무해(실측). rc=13은 집계 오탐 — 판정은
  per_episode 행 수. drawer-L 반전 판은 fail-loud로 자동 결손.

## 3. 잔여 (미지시 — 사용자 결정 대기)

1. **srv48 몫 27판**(coffee·dish·bread-w1) — v2판·v4판 모두 미실행(자리 대기). coffee는
   v4 detector δ 부재라 dish·bread-w1만 유의미.
2. **α=0.05 스윕**: v4 ckpt에 저장돼 있어 serve `--failure-alpha 0.05`만으로 setm 파손
   9건 재시험 가능(재학습 불요).
3. condg v4 위축 원인(미등록 cluster skip-dose) 진단 — sidecar `perstep_gate_skipped` 집계.
4. FP 억제 규칙(#11: 연속 M record 발화 시만 개입)·상시-재샘플 대조(#13).
5. drawer-L instruction 반전·coffee δ 구조 문제.
6. 정리 대기: 진단 pkl ~12GB(`og_ps_pilot_cap{,_jug}`)+`og_ps_pilot_cap` 계열, 원격
   segA_v4 ~120G, `detector_v4/cluster-k8_bad/`. 라운드 문서(docs/steering/48) 미작성.
7. 아티팩트 영상을 v4판 구제 사례로 갱신(렌더러는 per-step 대응 완료 상태).

## 4. 새 세션 프롬프트 (복붙용)

"main checkout ~/pkt_ws/temporal_vla (branch feat/rs-steer-v4)에서 per-step cluster
steering 라운드 후속을 진행해줘. 정본 = docs/steering/47 + docs/collab/
handoff_20260831_perstep_cluster.md — §1 판정(재론 금지)과 §2 좌표를 그대로 쓰고,
§3 잔여 중 지시받은 것부터. latch 시대 판정 인용 금지."
