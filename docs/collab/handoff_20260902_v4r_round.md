# Handoff — 중추(전체 파이프) 세션: rsN best-of-N · v4sb · v4r 재수집 라운드 (2026-09-02)

> 이 문서가 [`handoff_20260831_perstep_cluster.md`](handoff_20260831_perstep_cluster.md)를
> **대체**한다. 구 문서는 per-step 게이팅·cluster phase·v4 정렬 라운드(8-31까지)의 기록으로만
> 남는다 — §1-1~1-4 판정은 유효하나 최종표는 아래 §1-6 각주(noise-0 결손)를 함께 읽을 것.

## 0. 이 세션의 역할 (불변 — 구 문서 §0 승계)

**중추 세션** = SAFE detector(언제) + phase 판정(어디서) + steering 연산자(무엇으로)를
per-step 파이프 하나로 묶어 end-to-end 효과를 측정하고, 병목을 파트 담당 세션에 발주하고,
개선분을 재통합해 다시 돌리는 루프. 직접 = 배선·eval 실행·집계·판정. 위임 = 파트 내부 설계.

**협업 세션** (ListAgents → SendMessage): `연산자 설계`(setM·LLR 채점기 fit, 승준 배터리
보유) · `Steering 고찰`(설계 비판·위약/순환 논증) · `전체 파이프라인 dashboard`(포크 관찰자 —
설계 산출·진단 발주, GPU 안 만짐) · `action phase` · `시나리오 구체화` · `데이터 추가 수집`.
원격 CPU = remote-compute 에이전트(승준). GPU eval = 중추가 직접.

**용어** (구 문서 §0 사전에 추가되는 것만): **rsN_llr** = 발화 record에서 denoise noise 후보
N(=8)개를 DiT-only로 재추첨 → AE16 LLR 채점기로 최선 선택(critic형, 학습無). **rs1/reseed** =
후보 1개(=재추첨 1회). **fallback=reseed** = setm arm이 미등록 phase에서 발화하면 무개입
대신 재추첨 1회로 대체 개입. **v4sb** = 대상 scene 성공 배제 pool 규약(success-blind).
**v4r** = 대상 scene 25판을 eval 파이프(무개입 replay+캡처 ON)로 **재수집**한 정본.
**GT arm / ck8 arm** = setm의 phase 소스가 GT labeler / cluster k8.

## 1. 이번 세션 시도 → 결과 (시간순)

### 1-1. rsN best-of-N 연산자 배선 (완료, 커밋 e202c30~21ad51f)
- 연산자 설계 발주 스펙: 발화 시 N=8 재추첨 → 후보별 L12 h 캡처 → LLR(AE16·scene
  중심화·클래스 가우시안) argmin, OOD(5pct) 기각, 전부 기각 시 후보 0.
- 배선: serve `rsn_llr/rsn_rand` op(`_run_resample_gate`), `src/failure_online/llr_scorer.py`
  (scene-키 NPZ 계약 = docstring 단일 출처), 클라 `--perstep-n`, 러너 `ps_rsn_*` arm +
  `LLR_BUNDLE/LLR_SCENE/PERSTEP_N`. 후보별 llr·entry·log_s/log_f·기각·ms 로깅.
- 스모크에서 잡은 결함 3건: ① `vla_client.py` 매핑 누락(cand 로깅 유실) ② fallback을
  gate_skipped에 기록→applied_count 오집계(→`perstep_llr_fallback` 분리) ③ **"현재 cluster"
  기준 채점은 전 케이스 fallback 퇴화** — 발화 시점 cluster(후반 stall)와 등록 entry(초반
  창)가 전면 불일치 → **후보-기준 최근접 entry 채점(`score_nearest`)** 으로 개정, 발동=SAFE
  발화만. ④ 그래도 후보 전원 OOD → 연산자 측 ood_lo 스케일 버그(가우시안 상수항 −14.7)
  + 진성 격리 둘 다 → 발화-분포 재보정(ood_lo=실패 후반절반 5pct). 재보정 후 채점 실발생
  (비후보0 선택 115/131, 기각 5%).
- 사용자 지시로 arm 개편: rsN_rand 제외(≡rs1 합의), setm GT/ck8 2종 추가(미등록 phase는
  reseed fallback), 2차 pass 타이밍 로깅(rerun_ms·cand_ms). **β=1.5 무효 판정**("β>1은
  이상한 activation") → 스윕 전환. 그 라운드(og_rsn_expand)는 사용자 "다 멈춰"로 중단 —
  완료분(srv reseed/rsn_llr, `*_b15_invalid`)은 보존만, 판정 안 함.

### 1-2. 진단 3종 (Steering 고찰 제안, 기존 v4 정렬판 로그로 실측)
- **구제 셀 겹침**: reseed 구제 5셀 ⊂ setm 구제 6셀(공통 5/5, condg 유일 구제도 공통) →
  구제는 **경계셀 속성**이지 개입 내용이 아님. setm "최초 구제" 승격 보류(위약 관문 전).
- **hazard**: 구제 판 대부분이 수십 회 재시도 끝(marsh_s3만 4회). 비구제 판은 100+회에도 0
  = per-retry p≈0 이질 혼합 → 1−(1−p)^K 산수 폐기, **rsN_llr 판정 지표 재정의**: ①구제까지
  재시도 예산 곡선(A 대비) ②A-비구제 셀 전환(×3 재현 규칙) ③FP 파손 비회귀. 기존
  paired 표는 부표 존치.
- **triage**: 구제셀은 조기 발화(1st rec 22 vs 39)·비포화 score(0.84 vs 0.96). OvenRack
  (rec0 발화·score .98)·jug(정발화 rec50)는 전판 비구제 — "포화 score = 가망 없음"의
  온라인 신호(연산 배분 용도, 규칙화는 이르다).
- 캡처 hook 가설 **기각**: 반전 4판×2머신 = 캡처 ON에서도 전부 성공, 머신 간 steps bit 일치.
  GPU·머신도 무관(70/70). 반전의 실체 = **구 수집 rollout만 다른 세계**.

### 1-3. v4sb 설계(포크 세션 산출, 사용자 확정) → v4r 재수집 전환
- v4sb: instruction별 balance scene 1개(bread s1·drawer-L s4·marsh s3·candle s3·jug s4·
  dish s4·oven s4), fit pool = 타 4 scene 전체 + 대상 scene **실패만**, setM은 scene 잔차화
  후 instruction 단위, β 0.6~1.0, α=0.1, arm 순서 setm_gt→reseed→setm_ck8→llr.
  v4sb 산출물(detector_v4sb·ae_v4sb·instr_setm_v4sb_*·rsn_llr_reg_pool)은 구 라벨 기반이라
  **보존만, 미사용**.
- **사용자 결정(원인 조사 종료)**: 대상 scene 25판을 **eval 파이프 그대로(캡처 ON 무개입
  replay) 재수집**해 승준 데이터를 교체, 그걸로 detector/phase/연산자 재생성, β sweep.
  단 **eval도 같은 7-layer 캡처 hook을 켜되 activation은 저장·전송하지 않는다**
  (`DETECTOR_LAYERS=0,2,4,8,10,12,15`, 클라 `--no-features` 유지).
- 재수집 결과 **160판/실패 60** (jug 22·oven 17·candle 8·marsh 6·bread 5·dish 2·drawer-L 0).
  **drawer 방향은 지터 k가 재추첨** — s4의 k2/kbase/k6은 실제 right(fail-loud 제외), left
  셀은 전승이라 drawer-L은 eval 자연 탈락. 정본 = `v4r_labels.tsv`(160) · `v4r_eval.tsv`(60).
- 재학습 배터리(승준 ~85분): segA_v4r(대상 scene record를 신규 캡처·실패만으로 교체) →
  AE/KMeans k8 → segA_v4r_ck8 + dump-labels → detector_v4r(α 3종, 7/7 seed0). drawer-L은
  대상 scene 소멸로 2-scene 학습(결과표 명시). HDD CRC 손상 1건 재생성.
- 연산자 재fit(연산자 설계): instr_setm_v4r_gt(6 task) · instr_setm_v4r_ck8(oven 0 phase) ·
  rsn_llr_reg_v4r(등록 18셀, oven 0 → oven은 setm_gt·reseed만). NPZ 포맷 = **plain
  `setpoint`**(seg 아님) — 러너 화이트리스트 확장(c42ea61).

### 1-4. ★v4r β sweep 최종표 (60판 × 12 arm, α=0.1, N8, 구제 = 재수집 실패 → 성공)

| arm | 구제 | 주도 |
|---|---|---|
| reseed(rs1) | 3/60 (5%) | oven1·jug1·marsh1 |
| **rsn_llr** | **5/43 (12%)** | **candle 3/8**·bread1·marsh1 |
| setm_gt β0.6 / 0.7 / 0.8 / 0.9 / 1.0 | 5/59 · 2/58 · 2/59 · 1/60 · 5/60 | 0.6=bread2·oven2, 1.0=oven2·jug2 |
| setm_ck8 β0.6 / 0.7 / 0.8 / 0.9 / 1.0 | 3/43 · 3/42 · 1/43 · 2/43 · **5/43** | 1.0=marsh2·bread2 |

분모: 전 arm 60, **ck8·llr = 43 = 60 − oven 17**(연산자 부재 arm은 identity라 구조 제외).
59/58/42는 kanu 완강 실패(3회 재시도에도 rc=13, oven·marsh 판) 결손 — 미조사.
- 판독: ① 재수집 정본 프레임의 구제율은 최고 12% — 이전 17~21%는 라벨 반전 거품 포함.
  ② jug(22)·oven(17)은 전 arm 0~2 = 깊은 실패는 per-step 개입으로 거의 안 뒤집힘(hazard
  정합). ③ rsn_llr > reseed(공통 43판: 5 vs 2, candle 집중) — 채점기 순기여의 첫 양의
  신호, 표본 작음. ④ setm β 비단조(양끝 > 중간) — 판별력 부족이 더 큼. ⑤ 파손 축 없음(전
  판 base 실패; 성공 판 arm 투입은 사용자 미결). ⑥ dish 2·bread 5 표본 딱지.
- **가드 실증**: 첫 판에서 setm_gt 실적용 7/발화 16(GT phase 히트), rsn_llr 비후보0 선택
  47/49·기각 0 — 두 처치 arm 모두 퇴화 없음 확인 후 완주.

### 1-5. ★혼합 세계 문제 (사용자 지적 — 미결)
fit pool의 **타 4 scene은 여전히 구 수집 세계**(detector 학습·CP 보정·cluster 어휘·setM
"성공 방향" 대부분). 대상 scene만 replay 세계. v4 정렬에서 분포 정렬만으로 파손이 반감된
전례상 이 불일치가 낮은 구제율의 병목일 개연성. 제안한 2단: ① sidecar 분해로 병목 판별
② 타 scene 전면 재수집(7 task×100판=700판, 3머신 ~5-6h)으로 fit pool 전체를 replay
세계로 통일 후 재생성. **사용자 결정 대기** — 단 jug/oven류 깊은 실패는 세계 정합해도
한계 가능성(hazard).

### 1-6. 과거 표 캐비앳
awk falsy-"0" 함정(§4-1)이 og_ck8v4·og_rsn 계열 오케스트레이터에도 있었으므로 **v4 정렬판
최종표(구 문서 §1-4)도 noise-0 판이 빠졌을 수 있다**. 재실행 불필요(v4r가 대체) — 인용 시 각주.

## 2. 파트별 상태 → 다음 발주 후보

| 파트 | 상태 | 후보 |
|---|---|---|
| detector | detector_v4r 7 task α3종(대상 실패=신규, 타 scene=구 세계) | 혼합 세계 해소(§1-5), 완강 실패 판 조사 |
| phase(cluster) | ae_v4r_k8 번들, dump-labels | 발화 위치≠등록 위치 문제는 후보-기준 채점으로 우회 — 근본 해법(후반 창 등록)은 v3 이후 |
| 연산자 | instr_setm_v4r_{gt,ck8}(plain setpoint), rsn_llr_reg_v4r 18셀 | oven ck8/llr 0 → setm·reseed만; jug 성공 가우시안 얇음; 외삽 깊이(cand_logs) 사후분석 |
| 수집 | v4r_collect 160판(승준 61G) | 타 scene 700판 재수집 여부(§1-5) |
| 시나리오 | base-재정박·재수집 정본 프레임 전달됨 | 파손 측정용 성공 판 arm 투입 결정 |

## 3. 자원·서버 규약 (갱신분)
- **kanu**: serve 상주 ~5.8GB → **GPU당 serve 2**(사용자 확정, 6슬롯=3GPU×2). 총 3 GPU 상한·
  타인 프로세스 GPU 금지는 불변. `docker restart lerobot`로 NVML 복구(4일 가동 시 재발).
- srv50 GPU1 / srv48 GPU2: host-conda serve **6 병렬**(사용자 지시 — 판 수가 적어도 6개
  동시). srv→승준 직송 가능(`ssh -p 11112` 직결 확인, tar 스트림). 머신 매칭: bread s1은
  kanu 20+worker1 5 혼재.
- 승준: `~/datasets/.../grid_phase_v4/{v4r_collect(61G), segA_v4r(24G), segA_v4r_ck8(24G),
  segA_v4sb·_ck8(50G, 보존만)}`, 잔여 ~73G. 원격 전용 스크립트 `~/workspace/temporal_vla/
  outputs/tmp/{v4sb,v4r}/`(git 밖).

## 4. 이번에 밟은 함정 (구 문서 §4에 추가)
1. **awk falsy-"0"**: `ns[key]=ns[key]?...`가 noise "0"을 거짓으로 봐 n0 판 무음 탈락 →
   `(key in ns)` 판정. **완주 후 매니페스트 대비 커버리지 감사 필수**(재발 2회).
2. **매니페스트 ktag**: v4 index `armsig`는 전 행 'base' — 지터는 `jitter_reset_idx`(99=base)
   로. `cell_si`는 평탄 셀코드(scene*100+k), per_episode `scene_idx`는 base scene — 혼용 금지.
3. **캡처 수집 집계 공백**: 캡처 모드는 grid 좌표로만 저장 → raw_rollouts sidecar 없어
   collect_results INCOMPLETE → 러너 재시도가 자기 pkl과 덮어쓰기 tripwire 충돌. 이중기록
   fix(7c1a11f). 같은 OUT_ROOT에 job 겹치면 같은 증상.
4. **serve 즉사 = 30분 대기**: 기동 실패(NPZ 포맷 불일치 등)여도 러너 부팅 루프가
   SERVE_BOOT_TRIES 소진까지 기다린다 → "행업"처럼 보임. serve 로그 `startup failed` 확인.
5. NPZ 포맷: instr-단위 fit은 plain `setpoint`(키 alpha0_s/alpha0_v_steer) — 러너
   `STEER_OP=setpoint`. seg판은 `setpoint_seg`.
6. drawer는 **지터 k가 좌/우 방향을 재추첨** — instruction 프레임에 넣으려면 셀별 lang 검증.
7. pkill self-match·ssh setsid 붙들림·`local a=$1 b=..$a`·NVML 상실은 구 문서 그대로 재발.
8. 스모크 스크립트는 오케스트레이터와 **같은 env를 복사**할 것 — gt arm에 cluster 번들을
   넘겨 fallback만 도는 등 배선 차이로 판정을 오독하기 쉽다.

## 5. 좌표 인벤토리 (v4r 계열)
- 매니페스트: `outputs/steer/online_pipe/manifests/{v4r_collect.tsv(160), v4r_labels.tsv(160,
  정본), v4r_eval.tsv(60), rsn_llr_eval.tsv(구), v4sb_eval.tsv(구·ktag 버그 있음)}`.
- 번들/detector: `outputs/analysis/grid_phase/{ae_v4r_k8/ae_bundle_v4r_k8.npz,
  detector_v4r/cluster-k8/}` (v4sb 판은 `*_v4sb*`, 구 `ae_v4_k8`·`detector_v4`).
- 연산자: `outputs/steer/online_pipe_v4_pilot/{instr_setm_v4r_gt,instr_setm_v4r_ck8,
  rsn_llr_reg_v4r}/<task>/…` (러너용 심링크 루트 `instr_roots/<task>/<task>/`).
  LLR 계약 = `src/failure_online/llr_scorer.py` docstring.
- eval: `outputs/eval/robocasa/groot_n15/og_v4r_expand/<arm>/<case>/`(3머신, srv 몫은 각
  서버 동일 경로; 회수본 `outputs/tmp/v4r_results/`). 재수집 `og_v4r_collect/`(pkl 포함,
  승준 복제됨). 구 라운드 `og_rsn_expand`, `og_ck8v4_expand{,_srv50}`, 진단
  `og_capture_probe`(pkl 삭제 대상).
- 오케스트레이터(untracked `outputs/tmp/`): `og_v4r_expand_{kanu,srv}.sh`(6슬롯·행수 기반
  skip·defer/skip-struct), `og_v4r_collect.sh`, `og_capture_probe.sh`, `og_v4r_smoke.sh`(srv50).
  집계 = 인라인 python(per_episode glob → arm×(case,noise) 성공표) — 스크립트化 잔여.
- 서버 환경: srv50/48 repo `~/pkt_ws/temporal_vla`(git pull), 자산은 tar 반입.

## 6. 잔여 (사용자 결정 대기 순)
1. **혼합 세계 해소** — sidecar 분해 먼저 vs 타 scene 700판 재수집(§1-5).
2. 파손 축: 대상 scene 성공 판 일부를 arm에 투입할지(포크 미결).
3. sidecar 심층 분해: LLR cand_logs(외삽 깊이×선택 품질), 예산 곡선, setm 실적용/fallback
   비율별 구제 분해, 타이밍(rerun_ms) 표.
4. kanu 완강 실패 ~5 run(oven) 원인.
5. 라운드 문서 `docs/steering/48` 작성(이 핸드오프 §1이 초안), RESULTS.md 원장 갱신은 이번에 반영.
6. 정리: `og_capture_probe`·`og_v4r_smoke` pkl 삭제, `*_b15_invalid`·v4sb 산출물 존치 여부.
7. 집계 스크립트化, serve 즉사 조기 중단(부팅 루프) 개선.

## 7. 새 세션 프롬프트
"main checkout ~/pkt_ws/temporal_vla (branch feat/rs-steer-v4)에서 **중추(전체 파이프)
세션**을 이어받아줘. 정본 = docs/collab/handoff_20260902_v4r_round.md (§0 역할 그대로;
구 handoff_20260831은 배경). 게이팅 설계는 docs/steering/47. §1 판정 재론 금지, §6 잔여 1번
(혼합 세계 해소)부터 사용자 지시 받아 진행. 협업 세션은 ListAgents로 확인."
