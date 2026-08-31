# Handoff — 중추(전체 파이프) 세션: per-step 게이팅·cluster phase·v4 정렬 라운드 (2026-08-31)

## 0. 이 세션의 역할 — 반드시 이 정체성으로 이어받을 것

이 세션은 **중추 세션**이다: SAFE failure detector(언제 개입) + phase 판정(어디에/어떤
상태에) + steering 연산자(무엇으로)를 **하나의 per-step 파이프로 묶어 end-to-end 효과를
측정**하고, 병목을 파트별로 진단해 **각 파트 담당 세션에 개선을 요청**하고, 개선분을
다시 모아 전체를 재실행하는 루프를 돈다. 직접 하는 것 = 파이프 배선·eval 실행·집계·판정.
위임하는 것 = 파트 내부 설계 지식(담당 세션에 SendMessage로 문의/요청).

**협업 지형** (ListAgents로 확인, 이름으로 SendMessage):
- `action phase` 세션 — activation cluster/phase 판정 담당. 이번 라운드에서 ae_cluster
  k8 이식 세부를 문의해 받았음(AE 미저장 갭, 전처리식, 온라인 판정 수식).
- `시나리오 구체화` — hotfix 시나리오·논문 프레이밍(45 정본) 담당. **어떤 상황에서 뭘
  hotfix할지는 여기 소관** — 중추는 측정 결과를 넘기고 스펙 개정 필요분을 보고한다.
- `데이터 추가 수집` — 수집 라운드 담당(v4 지터 수집이 여기 산출).
- 원격 compute(승준 노드) = remote-compute 에이전트, GPU eval = 중추가 직접.

**프로젝트 목적**: VLA의 반복 실패를 백본 무학습으로 **가볍게 hotfix**한다 — 배포 로그로
detector+연산자를 자동 fit해 per-step으로 감지·개입. 이 세션은 그 파이프의 실효성을
측정하는 자리다. 방향 정본 = docs/steering/RESEARCH_DIRECTION.md, 시나리오 정본 = 45,
게이팅 설계 정본 = **47_perstep_gating_pipeline.md**. latch 시대 문서·판정은 폐기(재론 금지).

## 1. 이번 세션에서 시도한 것 → 결과 (시간순, 판정 포함)

### 1-1. latch 폐기 → per-step 게이팅 재설계·구현 (완료)
- 동기: 사용자 정식화 — LSTM detector는 개입 후 입력이 바뀌므로 상태가 실행된 세계를
  따라야 함 + 연산자가 detector와 같은 공간(DiT L12)을 밀기 때문에 steer된 값을 detector에
  먹이면 **점수 하락이 개입의 정의**가 되는 순환(Goodhart). latch(발화 후 상시 적용)는
  의도가 아니었음이 확인돼 관련 문서·판정 전부 폐기.
- 설계(47): 매 record 1차 **무개입** pass로 y_t 판정 → 발화 시 **DiT-only 2차 pass**
  (VLM backbone 캐시 재사용, 추론 비용 발화 record만 +1)로 action 교체 → detector는
  (h,c) 롤백 후 2차 활성화 x_t′로 재step(**h는 실행된 세계를 따름**, y_t′ 별도 기록).
  y_t−y_t′ = read-erasure 지표(연산자가 판독을 지우는 정도; setM만 ≫0).
- 검증: 스모크 5/5 — 배관 동치 max|Δaction|=0.0(144 record), 발화 재현, 개입 인과(개입
  다음 record부터 활성화 분기), y′ 기록, 2회 실행 bit 재현.
- 첫 실증: GT-phase 파일럿에서 **marshmallow ps_reseed 구제** — 발화 13회 만에 detector
  자연 침묵·조기 성공 = "개입→다음 step 자연 활성화로 재판정→회복 시 침묵" 폐루프 작동.

### 1-2. GT phase → cluster phase 전환 (완료)
- 동기: action phase 세션 실측 "GT event-labeler phase가 활성화 동역학과 비정합".
- 정본 채택: **ae_cluster k8** (raw1536 → 전역 mean-center+스칼라 std → AE 1536→256→256→16
  → per-task KMeans k8). AE ckpt가 저장 안 되던 갭을 `--export-bundle`로 봉합(번들 =
  mu/std + encoder state + task별 centers = serve·fit·검증의 단일 출처).
- 사전 진단 게이트 3종(비판 검증용)을 만들어 통과 확인 후 진행: 순환 누수(실패-전용
  cluster의 성공 표본), OOD 점유 붕괴, **절제 후 길이 confound 부활**(cluster는 진동해
  dwell-cap 절제가 fail 꼬리를 덜 자름 — 지표 부풀림 없음을 GT 대조 detector로 확인).
- 파이프 배선: serve가 1차 pass 활성화에서 cluster 자체 판정(GT POST 무시), 연산자
  fit은 `--cluster-bundle` 어댑터로 phases 치환(GT는 phases_gt로 병행 보존).
- cluster 파일럿 성과: **setM 최초 구제**(marshmallow, 실개입 3회 — exp2 이래 처음).

### 1-3. ★수집 라벨 비신뢰 발견 → base-재정박 프레임 (판정 확정)
- 51케이스 확장에서 ps_base(무개입)가 수집-실패 판의 ~55%에서 성공 — **개입 탓이 아니라
  v4 수집 시점 라벨이 경계 판에서 비결정**(고부하 병렬 수집 추정). 반면 **replay 자체는
  완전 결정적**: base 2회 36판 불일치 0, v2판↔v4판 base 교차 65/65 일치, seed 좌표
  전수 일치 확인.
- → 판정 프레임을 "수집 실패 구제"에서 **base-replay 재정박 paired**(구제=base실패→성공,
  파손=base성공→실패)로 확정. 시나리오(45)의 "재발 실패" 정의에 영향 — 시나리오 세션에
  전달 필요(§3).

### 1-4. ★detector 분포 정렬(v2→v4)이 승부수 (최종 판정표)
- 문제 발견 경로: cluster·detector를 v2 grid로 학습해 v4 지터 격자를 eval → CP 밴드
  동작점이 어긋나 성공 판에서 FP 폭주 → 파손이 구제의 2~4배(v2판). detector 재학습만으로
  task 동작점이 요동한다는 것도 실측(bread FPR 0.06→0.67 = 실전 발화 1→117회).
- v4 segA를 새로 추출(추출기를 지터-인식으로 패치)해 cluster·detector를 **eval과 같은
  분포로 재학습** → 최종표 (base-paired, drawer-L 제외):

  | arm | **v4 정렬판** 구제/파손 (68 pair) | v2 학습판 (65 pair) |
  |---|---|---|
  | ps_reseed | **5/29(17%) · 5/37(14%)** | 2/28 · 10/34 |
  | ps_setm | **6/29(21%) · 9/34** | 3/29 · 12/33 |
  | ps_condg | 1/26 · 4/20 | 3/27 · 6/22 |

- 판정: 구제 ~2배, 파손 절반. reseed 순효과 균형 도달, setM 구제율 최고(bread2·marsh2·
  candle2). **파손의 주범 = detector 동작점 불일치(FP)로 확정** — "정렬이 load-bearing".
- v4판 detector ckpt에 **α 0.05/0.1/0.2 3종 저장**(FP 레버, 재학습 없이 serve 인자로 스윕 가능).

## 2. 파트별 상태 → 개선 요청 후보 (다음 루프에서 담당 세션에 발주할 것)

| 파트 | 현재 상태 | 개선 요청 후보 |
|---|---|---|
| **SAFE detector** | v4 정렬판 8 task(α3종). 잔여 파손 = FP | ① α0.05 스윕(즉시, 중추가 직접 가능) ② FP 억제 규칙(연속 M record 발화 시만 개입) ③ 동작점 보정 표준화: "detector는 반드시 eval 분포로 CP 보정" 규약화 |
| **phase(cluster)** — action phase 세션 | k8 번들 v4판. 한계: rack 진행/역행 축 못 자름(0.73 vs probe 0.905), 실패-전용 cluster에 성공표본 결핍 | ① 방향 판독기 하이브리드(rack류) ② k 상향(margin은 k16~24 ≫ k8 실측) ③ 미등록 cluster fallback 설계 |
| **연산자** | setM 첫 구제·per-cluster fit은 등록 희소(케이스당 2~3/8 cluster, 전부 소표본 강제등록). condg는 v4에서 위축(skip-dose 진단 필요) | ① 미등록 cluster 대책(이웃 차용/global fallback) ② condg skip 원인 진단 ③ β·적용 call 스윕 |
| **수집** — 데이터 수집 세션 | v4 라벨이 경계 판에서 비신뢰(§1-3) | ① 수집 규약에 "저부하 수집 또는 수집 직후 base replay 재검증" 추가 요청 ② 성패 라벨은 replay 재검증본을 정본으로 |
| **시나리오** — 시나리오 구체화 세션 | 45의 "재발 실패" 분모 정의가 수집 라벨 전제 | base-재정박 프레임 반영한 스펙 개정 논의 |

## 3. 자원·서버 규약 (이 세션이 실측·확립한 것)

- **kanu(로컬)**: A4000 16GB×8, **serve 1개/GPU**(GR00T ~11.6GB). GPU 0-3 동료 예약
  경향·수시 변동 — **발사 직전 compute-apps 소유자 확인, 타인 프로세스(443MiB 상주
  포함) 있는 GPU 금지, 총 3 GPU 상한**. eval serve는 lerobot 컨테이너(docker exec -d).
- **srv50 = `AISem_50_junhyeong`(worker2)** / **srv48 = `AISem_48_junhyeong`(worker1)**:
  A100 80GB×4. **serve = host conda** `~/miniconda3/envs/lerobot_050_groot/bin/python`
  + `SERVE_PYTHONPATH=~/pkt_ws/temporal_vla/lerobot/src`, **serve 6/GPU·빈 GPU만**.
  repo = `~junhyeong/pkt_ws/temporal_vla`(git pull 가능; 연산자 NPZ·번들·ckpt는 git에
  없으니 tar로 반입). 머신 매칭: eval 판은 수집 머신에서(v4_expand_eval.tsv의 machine 열).
- **승준 원격**(remote_compute.sh, kimseungjun@166.104.146.37:11112): 대용량 shard·pkl
  전부 여기. CPU 8코어(스레드 cap 8)·RAM 26G — 무거운 job 병렬 금지. 코드는 git 브랜치
  동기화만(scp 금지), 장기 체인은 setsid detach + sentinel 파일 + 메인 세션 폴링.

## 4. 이번에 밟은 문제·함정 (다음 세션은 그대로 재사용할 것)

1. **docker NVML 상실**: 장기 가동 컨테이너가 GPU를 잃음(`Failed to initialize NVML`)
   → serve가 CPU로 떠서 FlashAttention 에러로 위장 사망 → `docker restart lerobot`.
   kanu에서 실제 발생, srv도 대비.
2. **pkill/pgrep self-match**: 명령줄에 대상 문자열이 들어가면 자기 자신을 죽임(로컬
   compound·ssh 원격 한 줄 명령 모두 당함) — kill과 발사는 **별도 ssh/커맨드로 분리**.
3. `set -u`에서 `local a=$1 b=...$a...` 한 줄 선언 → $a 미정의로 빈 값(무음 폴백 사고)
   — 줄 분리. 러너 파일은 실행 중 수정 금지(기존 함정 재확인).
4. **ssh 원격 setsid는 세션이 붙들림** — `ssh -f` 또는 발사 후 별도 ssh로 확인.
5. v2/v4 산출물은 **접미로 완전 분리**(ae_v2_k8/ae_v4_k8, detector_v2/detector_v4,
   case_*_ck8/ck8v4, og_ck8_*/og_ck8v4_*) — 덮어쓰기 금지 규약 유지.
6. v4 shard 추출은 지터-인식 extract_grid_matrix(c409d9d) 필수 — 좌표 중복·bread
   kanu∪worker1 union. v4는 scene 5개라 detector 분할은 3/1/1(기본 6/2/2는 무음 퇴화).
7. 집계 rc=13은 오탐(판정=per_episode 행 수), rc=12=NPZ 미등록(연산자 없음 케이스),
   drawer-L 반전 판은 fail-loud 자동 결손. EVAL_NOISES 다중 배칭은 결정성에 무해(실측).
8. 구조 제외 task: apple(v4 실패 0), coffee(성공 s2 편중 → CP 밴드 증명적 불가),
   drawer-L(replay instruction 좌우 반전 — 미해결 조사 대상).

## 5. 좌표 인벤토리

- 번들: `outputs/analysis/grid_phase/ae_v4_k8/ae_bundle_v4_k8.npz` (v2: `ae_v2_k8/`).
  판정기 `src/failure_online/cluster_phase.py`; fit 어댑터 = fit_setm/fit_cond_guidance
  `--cluster-bundle/--cluster-task`.
- detector: `outputs/analysis/grid_phase/detector_v4/{cluster-k8,phase-gt-v4}/`(8 task,
  excluded_tasks.tsv·adopted_split.tsv 동봉; v2판 detector_v2/).
- 연산자: `outputs/steer/online_pipe_v4_pilot/<case>/{case_setm_ck8v4,case_ck8v4}`
  (51케이스; condg layer = layer_choice.txt).
- eval: `outputs/eval/robocasa/groot_n15/og_ck8v4_expand{,_srv50}`(v4판),
  `og_ck8_expand{,_srv50}`(v2판), `og_ck8_pilot`, `og_ps_pilot`(GT). 판 목록 정본 =
  `outputs/steer/online_pipe/manifests/v4_expand_eval.tsv`(118판·machine 열), 케이스 =
  `v4_rescue_cases.tsv`, fit 매니페스트 = `v4_fit_all/`.
- 오케스트레이터(untracked): `outputs/tmp/og_ck8v4_expand_kanu.sh <GPU...>`(케이스 NPZ
  라우팅 `.roots/` 심링크·ps_base2 안정성 arm·멱등) / srv 기기 내 `og_ck8v4_expand_srv.sh`.
- serve/클라 계약: payload `perstep_gate{op,reseed_offset}`·`perstep_debug_rerun`; 응답
  `features.{failure_*,failure_score_post,perstep_fired/op/seed2/cluster,gate_skipped}`;
  클라 `--gated-steering-mode perstep --perstep-op {none,reseed,setm,condg}
  --perstep-cluster-phase`; 러너 env `CLUSTER_BUNDLE`·`DETECTOR_CKPT`.
- 영상: 렌더러 `scripts/analysis/grid_phase/render_activation_traj.py`(per-step 대응),
  아티팩트 https://claude.ai/code/artifact/4fbe7cf9-4c77-4230-bcd4-39e97ec03f1e (GT
  파일럿 기준 — v4판 미갱신).

## 6. 잔여 (사용자 결정 대기)

1. srv48 몫 27판(coffee·dish·bread-w1) — v2·v4판 모두 미실행(coffee는 v4 δ 부재).
2. α=0.05 스윕으로 setM 잔여 파손 재시험(serve 인자만, 재학습 불요).
3. condg v4 위축 진단(sidecar perstep_gate_skipped 집계) → 연산자 파트 발주.
4. FP 억제 규칙(#11)·상시-재샘플 대조(#13)·margin 자가평가(#8).
5. 파트 발주(§2 표) — 특히 수집 규약 개선·시나리오 스펙 개정 전달.
6. 정리: 진단 pkl ~12GB(og_ps_pilot_cap*), 원격 segA_v4 ~120G, detector_v4/cluster-k8_bad.
7. 라운드 문서 docs/steering/48 작성·아티팩트 v4판 갱신.

## 7. 새 세션 프롬프트 (복붙용)

"main checkout ~/pkt_ws/temporal_vla (branch feat/rs-steer-v4)에서 **중추(전체 파이프)
세션**을 이어받아줘. 역할·판정·좌표·함정의 정본 = docs/collab/
handoff_20260831_perstep_cluster.md (+게이팅 설계는 docs/steering/47). §0 역할 그대로 —
파이프 통합 측정이 본업이고 파트 개선은 담당 세션(action phase·시나리오 구체화·데이터
추가 수집)에 발주한다. §1 판정 재론 금지, §6 잔여 중 사용자 지시부터. latch 시대
판정 인용 금지."
