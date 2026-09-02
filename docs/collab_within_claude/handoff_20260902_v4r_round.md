> **2026-09-02 후속(v5 첫 셀 게이트)**: 여기서 관찰된 "수집 실패 → 무개입 replay 성공 59%" 의 원인은
> replay 가 `EP_META_DIR`(ep_meta JSON 을 `reset(seed)` 전에 주입) 로 돌아 **k 번째 지터 상태가 수집과
> 달랐기 때문**이다(같은 좌표에서 drawer 방향까지 바뀜). JSON 미로드 replay 는 수집과 bit 동일. 상세
> `handoff_20260902_grid_recollect_v5.md` §0.5.1. 이 문서의 v4r 라벨은 수집 좌표와 다른 판의 라벨이다.

# Handoff — 중추(전체 파이프) 세션 (2026-09-02)

> [`handoff_20260831_perstep_cluster.md`](handoff_20260831_perstep_cluster.md) 대체. 구 문서는 8-31까지 기록용.
> **데이터는 전부 다시 모을 예정** → v4/v4sb/v4r 수치는 참고만. 남는 건 파이프 코드·규약·함정.

## 0. 이 세션의 역할

**중추 세션** = SAFE detector(언제) + phase 판정(무엇을) + steering 연산자(어떻게)를
per-step(매 step 판단, 개입) 파이프 하나로 묶어 end-to-end 효과 측정 → 병목을 파트 세션에 발주 → 개선분 재통합 → 다시 돌림.
직접 = 배선·eval 실행·집계·판정. 위임 = 파트 내부 설계.

**협업 세션** (ListAgents → SendMessage)
- `연산자 설계`: setM·LLR 채점기 fit, 승준 배터리 보유
- `Steering 고찰`: 설계 비판·위약/순환 논증
- `전체 파이프라인 dashboard`: 포크 관찰자 — 설계 산출·진단 발주, 코드생성, process 실행 x
- `action phase`: phase detector
- `시나리오 구체화`: 연구 주제의 중심
- `데이터 추가 수집`: 사용데이터의 종류, 저장형태, 모델, 위치 관리
- 원격 CPU = remote-compute 에이전트(승준). GPU eval = 중추가 직접 각 서버(48, 50, kanu)에 명령 실행.
- 서버 할당 규칙: §3 + `.claude/skills/robocasa-steer-eval/SKILL.md`

**대상**: GR00T N1.5 × RoboCasa. scene 5 × noise 5 × 지터 k 5 = 125판/instruction (v4 grid). 실패 = timeout(144 record).

instructions
1. open the left drawer
2. Fully slide the oven rack out
3. Fully slide the top dishwasher rack out
4. **Pick Place** Counter To Cabinet ~~5~~→ 4**종** (~~apple~~, bread, candle, jug, marshmallow) apple은 100% 성공
5. ~~setupcoffee~~ 실패만 너무 많이 모임
6. ~~open the right drawer~~ 성공만 너무 많이 모임

**용어**
- record = 모델 inference 1회(5 env-step마다). 발화 = detector가 그 record에서 fail 알림. 개입 = 발화 record의 action을 2차 pass로 교체(1회성).
- SAFE detector = DiT L12 activation(1536) 받는 LSTM. α = 허용 FPR(0.05/0.1/0.2, ckpt에 셋 다 있음). **β는 별개** = 연산자 개입 강도.
- phase = 개입 슬롯 키. GT = 이벤트 라벨러 phase. ck8 = activation AE(1536→16) 위 per-task KMeans k8(c0~c7).
- setM = 성공 setpoint 쪽으로 activation 당김(phase별 NPZ). condg = 조건부 대조 guidance(이번 미사용). reseed = denoise noise 재추첨 1회.
- rsN_llr = 발화 시 noise 후보 8개 재추첨(DiT-only, VLM 캐시) → LLR 채점기로 선택. llr = log_f − log_s, 낮을수록 성공스러움 → argmin. OOD 기각 = max(log_s, log_f) < ood_lo.
- fallback=reseed = setm이 미등록 phase에서 발화하면 reseed로 대체 개입.
- 구제 = 무개입 replay 실패 → 개입 arm 성공. 파손 = 반대(이번 측정 x).
- sidecar = 판별 json(`raw_rollouts/.../task0--ep*--succ*.json`). record열에 failure_scores / perstep_fired / gate_fallback / cand_* / rerun_ms.
- 지터 k = 같은 scene seed에서 배치·관절 k회 재추첨. cell_si = scene×100+k (s3_k1 → 301, kbase → 399).
- v4sb = 대상 scene 성공 배제 pool. v4r = 대상 scene 25판을 eval 파이프(무개입 replay + 캡처 ON)로 재수집한 것.
- rc=13 = per_episode 행 수 미달. fail-loud = 조용히 no-op 하지 말고 죽이기.
- 승준 = 원격 CPU 노드(kimseungjun@166.104.146.37:11112, shard·pkl 보관). kanu = 로컬 A4000×8. srv50/48 = A100(ssh `AISem_50_junhyeong`/`AISem_48_junhyeong`, worker2/worker1).
- "재시도" 두 뜻: hazard = 발화 지속 시 매 record 재추첨 / 운영 = 러너 job 재실행.

## 1. 이번에 한 것 (시간순)

### 1-0. 반전 문제
v4 수집 때 "실패"인 판을 eval 파이프에서 무개입 replay → **59%(36/61) 성공**. replay 자체는 완전 결정적(2회 불일치 0, 머신·GPU·캡처 바꿔도 bit 일치).
→ 구 수집 rollout만 다른 세계. 원인 조사 종료 → 데이터 재수집 했다. 문제해결.

### 1-1. rsN best-of-N 배선 (커밋 e202c30~c42ea61)
- serve `rsn_llr` op(`_run_resample_gate`), `src/failure_online/llr_scorer.py`(NPZ 계약 = docstring), 클라 `--perstep-n/--perstep-fallback`, 러너 `ps_rsn_llr` arm + `LLR_BUNDLE/LLR_SCENE/PERSTEP_N/PERSTEP_FALLBACK`.
- 로깅: 후보별 llr·entry·log_s/log_f·기각, 2차 pass 시간(rerun_ms·cand_ms). 발화 record 개입 비용 ≈ 50ms/후보.
- 고친 것: ① vla_client 매핑 누락 ② fallback을 gate_skipped에 넣어 applied 오집계 ③ **"현재 cluster" 기준 채점 = 전부 fallback** (발화는 후반 stall cluster, 등록은 초반 창) → 후보-기준 최근접 entry 채점(`score_nearest`), 발동 = SAFE 발화만 ④ 후보 전원 OOD → 연산자 쪽 ood_lo 상수항 버그 + 진성 격리 → 발화-분포로 재보정.
- arm 개편(사용자): ~~rsN_rand~~(≡reseed), setm GT/ck8 2종 추가, ~~β1.5~~ → β sweep 0.6~1.0.

### 1-2. 진단 3종 (Steering 고찰 제안, v4 정렬판 로그)
- 구제 셀 겹침: reseed 구제 5셀 ⊂ setm 구제 6셀. **구제 = 경계셀 속성**, 연산자 내용 x.
- hazard: 구제는 수십 회 재추첨 끝. 비구제는 100+회에도 0. → 1−(1−p)^K 산수 폐기. rsN_llr 판정은 ① 재추첨 횟수 누적곡선(reseed 대비) ② reseed가 못 구한 셀 전환(×3 재현 시만) ③ FP 파손 비회귀.
- triage: 구제셀 = 조기 발화(rec 22 vs 39)·비포화 score(.84 vs .96). oven(rec0 발화·.98)·jug(rec50 정발화) = 전판 비구제. 포화 score = 가망 없음 신호(연산 배분용).

### 1-3. v4sb → v4r (요약만. 데이터 전부 재수집 예정)
- v4sb 설계(포크): instruction별 balance scene 1개, fit pool = 타 4 scene + 대상 scene 실패만, setM은 scene 잔차화 후 instruction 단위, α 0.1, β 0.6~1.0.
- v4r: 대상 scene 25판 재수집(캡처 ON 무개입 replay). **eval도 같은 7-layer 캡처 hook 켜고 activation은 저장 x** (`DETECTOR_LAYERS=0,2,4,8,10,12,15`, 클라 `--no-features`).
- 175판 중 drawer-L 15판 제외 → 160판 / 실패 60. **drawer는 지터 k가 좌/우를 재추첨** (k2/kbase/k6 = right). left는 전승 → drawer-L eval 대상 없음.
- 재생성: detector_v4r(7 task, α 3종) · ae_v4r_k8 · instr_setm_v4r_gt(6) · instr_setm_v4r_ck8(5, oven 0) · rsn_llr_reg_v4r(18셀, oven 0). NPZ = plain `setpoint`.
- 가드: 첫 판 sidecar에서 setm 실적용 7/발화 16, llr 비후보0 선택 47/49 확인 후 완주.

### 1-4. v4r β sweep 결과 (참고용)
60판 × 12 arm, α 0.1, N8. 구제 = 재수집 실패 → 성공.

| arm | 구제 |
|---|---|
| reseed | 3/60 (oven 빼면 2/43) |
| rsn_llr | 5/43 (candle 3/8) |
| setm_gt β0.6 / 0.7 / 0.8 / 0.9 / 1.0 | 5/59 · 2/58 · 2/59 · 1/60 · 5/60 |
| setm_ck8 β0.6 / 0.7 / 0.8 / 0.9 / 1.0 | 3/43 · 3/42 · 1/43 · 2/43 · 5/43 |

- ck8·llr 분모 43 = 60 − oven 17(연산자 없음). 59/58/42 = kanu 러너 완강 실패 5 run(미조사).
- 구제율 최고 12%. 이전 17~21%는 라벨 반전 거품.
- jug(22)·oven(17) = 전 arm 0~2. 깊은 실패는 per-step으로 안 뒤집힘.
- rsn_llr > reseed (43판: 5 vs 2). 표본 작음.
- setm β 비단조. 판별력 부족.
- 파손 축 없음(전판 실패 판).

### 1-5. 혼합 세계 → 전부 재수집
fit pool 타 4 scene은 아직 구 수집 세계(detector 학습·CP·cluster·setM 방향). 대상 scene만 replay 세계.
→ **데이터 전부 다시 모으기로 결정.** 다음 라운드는 재수집 완료 후.

### 1-6. 구 표 주의
awk "0" 함정(§4-1)이 og_ck8v4·og_rsn 오케스트레이터에도 있었음 → 구 문서 §1-4 표도 noise-0 판 빠졌을 수 있음. 인용 시 각주.

## 2. 파트별 상태

| 파트 | 상태 | 다음 |
|---|---|---|
| detector | detector_v4r 7 task α3종 | 재수집 데이터로 재학습 |
| phase(cluster) | ae_v4r_k8 | 발화 위치≠등록 위치는 후보-기준 채점으로 우회 중. 실패 후반 창 기준 재등록은 채점기 개정 과제 |
| 연산자 | instr_setm_v4r_{gt,ck8}, rsn_llr_reg_v4r 18셀 | 재fit. oven ck8/llr 0 → setm·reseed만 |
| 수집 | v4r_collect 160판(승준 61G) | 전체 재수집 |
| 시나리오 | 재수집 정본 프레임 전달됨 | 파손 측정용 성공 판 arm 넣을지 |

## 3. 서버 규약 (갱신)
- **kanu**(A4000 16GB): serve ~5.8GB → **GPU당 serve 2**(6슬롯 = 3GPU×2). 3 GPU 상한·타인 프로세스 GPU 금지. serve는 lerobot 컨테이너. 4일 넘게 켜두면 NVML 죽음 → `docker restart lerobot`.
- **srv50 GPU1 / srv48 GPU2**(A100): host-conda serve(`~/miniconda3/envs/lerobot_050_groot/bin/python`, `SERVE_PYTHONPATH=~/pkt_ws/temporal_vla/lerobot/src`). **GPU당 6 병렬**. srv → 승준 직송 됨(`ssh -p 11112`, tar 스트림). repo `~/pkt_ws/temporal_vla` git pull, 자산은 tar.
- **승준**: `~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase_v4/` 아래 v4r_collect(61G)·segA_v4r(24G)·segA_v4r_ck8(24G)·segA_v4sb·_ck8(50G, 보존만)·v4r_labels.tsv·instr_setm_v4r_*·rsn_llr_reg_v4r. 잔여 ~73G(700판이면 +55G). repo `~/workspace/temporal_vla`. 체인 스크립트 `~/workspace/temporal_vla/outputs/tmp/{v4sb,v4r}/run_*.sh`(git 밖, 로컬 outputs/tmp와 다른 파일).

## 4. 함정 (구 문서 §4에 추가)
1. **awk "0"**: `ns[key]=ns[key]?…` 가 noise "0"을 거짓 취급 → n0 판 무음 탈락(2회 재발). `(key in ns)` 로. **완주 후 매니페스트 대비 행수 감사 필수.**
2. 매니페스트 열: v4 index `armsig`는 전 행 'base'. 지터는 `jitter_reset_idx`(99=base). `cell_si`는 scene×100+k, per_episode `scene_idx`는 base scene. 섞지 말 것.
3. 캡처 수집 집계: grid 좌표로만 저장하면 raw_rollouts sidecar 없음 → INCOMPLETE → 러너 재시도가 자기 pkl과 덮어쓰기 tripwire 충돌. 이중기록 fix(7c1a11f). 같은 OUT_ROOT에 job 겹치면 같은 증상.
4. serve 즉사 = 30분 대기: 기동 실패(NPZ 포맷 등)여도 러너가 SERVE_BOOT_TRIES 다 기다림. `serve_<port>.log`에 `startup failed` 먼저 볼 것.
5. NPZ 포맷: instr-단위 fit = plain `setpoint`(alpha0_s/alpha0_v_steer) → 러너 `STEER_OP=setpoint`. seg판 = `setpoint_seg`.
6. drawer = 지터 k가 좌/우 재추첨. instruction 프레임 쓰려면 셀별 lang 확인.
7. pkill 자기매치·ssh setsid 붙들림·`local a=$1 b=..$a`·NVML = 구 문서 그대로 또 밟음.
8. 스모크는 오케스트레이터와 **같은 env 복사**. gt arm에 cluster 번들 넘기면 fallback만 돌아서 오판.

## 5. 좌표 + 커맨드 (v4r 계열)
- 매니페스트: `outputs/steer/online_pipe/manifests/{v4r_collect.tsv(160), v4r_labels.tsv(160, 정본), v4r_eval.tsv(60)}`. 구: `rsn_llr_eval.tsv`, `v4sb_eval.tsv`(ktag 버그).
- 번들/detector: `outputs/analysis/grid_phase/{ae_v4r_k8/ae_bundle_v4r_k8.npz, detector_v4r/cluster-k8/}`.
- 연산자: `outputs/steer/online_pipe_v4_pilot/{instr_setm_v4r_gt, instr_setm_v4r_ck8, rsn_llr_reg_v4r}/<task>/`. 러너 심링크 루트 `instr_roots/<task>/<task>/`. LLR 계약 = `src/failure_online/llr_scorer.py` docstring.
- eval: `outputs/eval/robocasa/groot_n15/og_v4r_expand/<arm>/<case>/`(3머신 분산, srv 회수본 `outputs/tmp/v4r_results/`). 재수집 `og_v4r_collect/`(pkl, 승준 복제됨). 구 라운드 `og_rsn_expand`, `og_ck8v4_expand{,_srv50}`. 진단 `og_capture_probe`(pkl 삭제 대상).
- 오케스트레이터(untracked `outputs/tmp/`): `og_v4r_expand_{kanu,srv}.sh`(6슬롯·행수 기반 skip·defer), `og_v4r_collect.sh`, `og_capture_probe.sh`, `og_v4r_smoke.sh`(srv50). 집계는 인라인 python(스크립트化 x).
  ```bash
  # eval 재개/보충 (멱등)
  setsid nohup bash outputs/tmp/og_v4r_expand_kanu.sh 4 5 6 > outputs/eval/robocasa/groot_n15/og_ps_smoke/logs/v4r_k.log 2>&1 < /dev/null &
  ssh -f AISem_50_junhyeong "setsid nohup bash ~/pkt_ws/temporal_vla/outputs/tmp/og_v4r_expand_srv.sh 1 </dev/null >/dev/null 2>&1"
  # 완주 = run.log 의 V4R_EXPAND_{kanu,worker1,worker2}_DONE. 결손 = rc=13 & per_episode 0행
  # 재수집(캡처 ON base) = og_v4r_collect.sh kanu 4 5 6 | srv 1|2  (v4r_collect.tsv)
  # 집계 = og_v4r_expand/<arm>/<case>/<task>/<arm>/per_episode.tsv 를 (case,noise)로 모아 success 합
  # sidecar = 같은 자리 raw_rollouts/<Task>/<slug>/task0--ep<N>--succ<0|1>.json
  ```

## 6. 남은 것
1. **데이터 전부 재수집** (데이터 수집 세션) → detector/phase/연산자 재생성 → β sweep 다시.
2. 파손 축: 대상 scene 성공 판도 arm에 넣을지.
3. sidecar 분해: cand_logs(외삽 깊이 × 선택 품질), 재추첨 누적곡선, setm 실적용/fallback별 구제, rerun_ms 표.
4. kanu 완강 실패 5 run(oven) 원인.
5. `docs/steering/48` 라운드 문서(이 §1이 초안). RESULTS.md엔 2라운드 행 넣어둠.
6. 정리: og_capture_probe·og_v4r_smoke pkl 삭제. `*_b15_invalid`·v4sb 산출물 둘지.
7. 집계 스크립트化. serve 즉사 시 부팅 루프 조기 중단.

## 7. 새 세션 프롬프트
"main checkout ~/pkt_ws/temporal_vla (branch feat/rs-steer-v4)에서 중추(전체 파이프) 세션 이어받아줘. 정본 = docs/collab_within_claude/handoff_20260902_v4r_round.md. 게이팅 설계 = docs/steering/47. §1 판정 재론 x. §6-1(데이터 재수집)부터. 협업 세션은 ListAgents."
