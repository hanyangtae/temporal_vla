# HANDOFF — scene-seed 매트릭스 진행 중 + 다음 로드맵 (새 세션 진입점)

> 2026-07-06 11:30 작성. 이 문서 하나로 새 세션이 **돌고 있는 실험을 이어받아 마무리**하고
> 다음 단계(A100 활용, scene 증가, β/α sweep, SAE)로 나아갈 수 있게 쓴다.
> 연구 플랜 원본: `~/.claude/plans/vla-reactive-acorn.md` (말미에 상태 포인터 있음).
> 폐기된 과거 라운드(구 라벨·in-sample 오염 실험)의 상세는 생략 — 결론만 §1.4에, 수치는 Notion/JSON에.

---

## 0. 한 줄

RoboCasa+GR00T N1.5에서 **대조 conceptor steering의 scene-seed 일반화 최종 매트릭스**(8 cell × ~19 arm)가
무인 파이프라인으로 **지금 돌고 있다**(bread 우선). 이것이 "고정 seed·no-SAE" 접근의 종결 실험이고,
이후 = A100 서버 활용 → scene 확장 / β·α sweep(체인 예약됨) → SAE.

---

## 1. 방법 기초 (새 세션이 알아야 할 정확한 정의)

### 1.0 수집·실행 스택 (rollout 1판이 도는 방식)
- **2-프로세스**: ① 모델 serve = `scripts/serve/lerobot.py` (lerobot 컨테이너 안에서 실행,
  `--profile configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml --collect --capture-vl
  --groot-dit-capture-layers 0,2,4,8,10,12,15` [+steering 옵션]), ② 수집 client =
  `scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py` (robocasa 컨테이너, MUJOCO_GL=egl,
  PYTHONPATH=Isaac-GR00T:robocasa:robosuite:repo). serve 1개 ≈ GPU 5.8GB (16GB 카드에 2개 동거 가능).
- **추론 표준**: action chunk **16 예측 / 5 step 실행 후 재계획** → get_action **1회 = feature record 1개 = env 5-step**.
  max 720 env-step, **timeout = 실패**, `step_success()` 신호 = 성공(조기 종료). 실패판 record 수 = 144.
- **캡처 내용(record당)**: DiT block-residual **[L=7(layer 0,2,4,8,10,12,15), K=4(denoise), D=1536]** fp16
  (`feature_kind=groot_n15_dit_block_residual_action_tokens_denoise`, action-token mean-pool) + VL `[2048]`
  (`vlln seq-meanpool`) + action/proprio + phase 라벨(1:1). fit 시 K축은 mean-pool → 주입점(D=1536)과 동일 공간.
- **gated steering 배선**: collector `--gated-steering`이 매 get_action 전 현재 phase를 serve `/steering_phase`로
  POST → serve가 layer별 hook의 M을 스위칭(등록 안 된 phase(terminal/wrong-grasp/global)는 identity).
- 산출 = (latent, phase 라벨, success/fail)이 **event/phase 기준으로 정렬된** 데이터셋. timestep이 아니라
  phase로 정렬하는 이유 = 시간 정렬의 길이 confound(실패=항상 timeout) 완화.
- ⚠️ **길이 confound는 phase 분리로 "해결"되지 않는다 — 오히려 다루기 어려워졌다** (주의 지속 필요):
  ① **phase 내 dwell 불균형** — 실패는 특정 phase(예: transport)에 훨씬 오래 머물러 fail-class R이 장기체류
  구간에 지배됨. dwell-matched 검사에서 겉보기 succ/fail 분리가 붕괴한 전례 있음(§1.4 결론의 근거 일부).
  ② phase-pool된 record 수가 클래스 간 크게 다름. 분리도·steer 효과 해석 시 dwell 통제 여부를 항상 명시할 것.

### 1.1 개입 연산자
- 클래스별 conceptor: **C = R(R + α⁻²I)⁻¹**, R = E[hhᵀ] (per-timestep record 상관행렬, rollout-pool 금지).
- 대조: **C_steer = C_succ ∧ ¬C_fail** (`src/conceptor/`).
- 주입: **M = (1−β)I + β·C_steer, h' = h·Mᵀ** — DiT `transformer_blocks[ℓ]` residual에 forward hook.
  기본 **multi-layer L4+L8+L12** (각 layer 자기 NPZ), **β=0.3 고정**.
- **α 자동 선택**: α∈{0.1,0.3,1,3,10} sweep → overlap(C_s,C_f)=tr(C_sC_f)/√(trC_s²·trC_f²) (Frobenius cosine)
  이 **밴드 [0.85,0.95]에 드는 최소 α**(없으면 경계 최근접). COAST 경험칙.
- ⚠️ **로더 함정**: serve가 `--steering-alpha` 미지정 시 NPZ **첫 키**를 로드 → 지금까지 전 결과는 **α=0.3 균일 적용**
  (선택 α 미반영; 내부 비교는 공정). 명시하려면 launcher env `STEER_ALPHA=1` (배선돼 있음).
- 고유값 실측: 어떤 fit(15~90, per-seed/cross)에서도 C_steer **λmax≈0.45–0.52, λ>0.5 = 0~1개/1536** = "거의 빈 연산자"
  (성공·실패가 구조 대부분 공유 → AND-NOT이 소거). β=0.3에서 방향별 차등 ≤ ~15%p.

### 1.2 라벨러 (6/7-phase, proximity 기반 causal — 현행 표준)
- `scripts/safe/groot_n16/{libero/event_phase_labeler.py, robocasa/collect/robocasa_event_labeler.py}`
  `make_robocasa_event_labeler(env, task, proximity_phases=True)` — **collector `--proximity-phases`로 켬**.
- phase = 현재 sim 상태의 순수 함수(비단조, drop시 복귀): reach(¬grasped∧far) / **grasp**(¬grasped∧near_obj,
  near_obj=¬OU.gripper_obj_far) / transport(grasped∧far-from-target) / **place**(grasped∧near_target;
  cabinet=obj_inside_of th 0.20, stove=xy<0.20) / insert-settle(placed) / terminal(released) /
  **wrong-grasp**(distractor 파지 — `env.objects`서 타깃·container 제외 발견, debounce).
- **wrong-grasp 데이터의 fit 처리** (정확한 semantics — `fit_phase_conceptor_n15.py gather_class_records`):
  wrong-grasp **라벨 구간의 record가 phase bin에서 제외**된다(fit group에 없음 → 라벨 분리로 자동 제외).
  distractor를 끝까지 쥔 채 timeout하는 전형적 실패에서는 라벨이 에피소드 끝까지 지속 →
  **사실상 wrong-grasp 발생 이후 데이터 전체가 날아간다**(놓으면 reach로 복귀해 이후 구간은 다시 포함).
  global bin만 사용자 결정으로 전 record 포함. gating 시 미등록 phase(wrong-grasp/terminal) → identity(무개입).
- 알려진 특성: **cabinet task는 reset부터 near_target=True라 transport가 place로 흡수**(사용자 승인됨).
- pkl 필드: `feature_phases`(record 1:1)·`phase_timeline`·`grasp/drop/wrong_grasp_steps`·`*_timeline`. 테스트 27 pass.

### 1.3 평가 프로토콜 (필수 준수)
- scenario_seed 고정 + **inference_seed = ep×1000** → (scene, ep)가 rollout을 **결정적으로 재현**.
- **fit ep0–59 / eval ep60–119** — steering eval은 fit에 쓴 inference_seed와 **반드시 분리**(in-sample rescue 함정).
- 판정 기준 2개: baseline 대비 + **re-roll null**(방향 없는 큰 섭동이면 steered SR ≈ 그 scene의 평균 SR로 수렴;
  flip 기대치 = 실패수×SR, 성공수×(1−SR)). per-seed flip(구제/파괴) 분해가 가장 정보량 큼.

### 1.4 확정된 결론 (상세 생략, 위치만)
3 scene(고정 seed) 최종: **raw conceptor는 모든 변형(fit 15–90·perm/gated·cross-instruction·라벨 엄밀화)에서
scene-일관 개선 없음** — bread만 국소 +, apple 전 조건 유의 해악, potato 바닥.
수치: `outputs/eval/robocasa/groot_n15/steer_eval/RESULTS_6p_final.json` + Notion(§6). 메모리 [[conceptor-steering-final-verdict]].

---

## 2. ★ 지금 돌고 있는 것 — scene-seed 매트릭스 (이어받아 마무리할 대상)

### 2.1 정의
- 마스터: `scripts/safe/groot_n15/robocasa/steer/master_final_scene2.sh` — **bread 우선 2-phase**, setsid detached.
  로그: `outputs/eval/robocasa/groot_n15/pipeline_final.log` (append; **최신 "FINAL2 START" 이후만** 봐야 함).
- **8 cell**: bread seed {100084, 300028, 300033, 400020} / apple seed {100074, 100050, 100084, 100104}
  (cell_id: `ppcc_bread`, `ppcc_bread_s300028`, …, `ppcs_apple_s100104`; task idx bread=5, apple=1).
- **arm** (전부 eval ep60–119, n=60, PROX=1): `ho_base` + {per-seed **ps**, 동일-instruction cross **xb/xa**,
  grand(8 cell) **gx**} × {perm, gated} × fit{**15,30,60**} + bread에만 layer 변형 `ho_gatedL4`, `ho_gatedL812`.
  arm 디렉토리: `steer_eval/<cell>/ho_{perm|gated}{ps|xb|xa|gx}{15|30|60}/raw_rollouts/...`
- conceptor: `phase_event_6p/analysis/final_{ps,xb,xa,gx}{15,30,60}/<cell>/{global,reach-to-object,grasp,transport,place,insert-settle}/dit_L*/conceptors.npz`
  (cross류는 `all/` 실체 + cell명 **상대 심링크** — 컨테이너 경로 호환).
- fit 데이터: `phase_event_6p/raw_rollouts/<task>/<cell>/` ep0–59 (6/7-phase, 결정적).
- GPU lane: bread84=GPU4(8470,8471), s400020=GPU7(8476,8477), s300028=GPU5(8474,8475), s300033=GPU6(8472,8473).
  apple phase도 4-lane 동일 구조 + 각 lane이 bread gx armset 후속. **GPU 0–3은 동료 — 사용 금지.**

### 2.2 진행 스냅샷 (2026-07-06 11:32)
- 마스터 PID **2574668** 생존(setsid 재부모화 주의: `$!`가 아니라 pgrep으로 확인).
- Phase B: S1 수집 완료(신규 bread 3 cell 60판씩), S2 fits 완료(`final_ps15/30/60` × 4 bread + `final_xb15/30/60`),
  **B-S3 진행 중**: bread84·s300028 armset 1/8 완료(ps15), 저SR 두 cell은 ps15 내부. CPU 경합(load 90대) 여파로 지연됨.
- baseline (완료): bread84 **47/60=.783** / s300028 **50/60=.833** / s300033 **25/60=.417** / s400020 **21/60=.350**
  — 난이도 스펙트럼 확보. perm15 완료 2건: 84 47/60(±0), 28 51/60(+.017) — 무효과 패턴 지속.
- 예정 완료: bread phase 후 `FINAL2_BREAD_DONE` + **bread 중간 집계→Notion 자동**
  (`aggregate_final_scene.py`, `BREAD_ONLY=1`); 이후 apple phase(A-S1 수집→fits(xa,gx)→A-S3) → 최종 집계+Notion+rsync
  → `FINAL2_ALL_DONE`.

### 2.3 모니터/운영 (새 세션이 할 일)
```bash
# 생존/진행
pgrep -f master_final ; tail -5 outputs/eval/robocasa/groot_n15/pipeline_final.log
for c in ppcc_bread ppcc_bread_s300028 ppcc_bread_s300033 ppcc_bread_s400020; do
  echo "$c: $(grep -c 'DONE ->' outputs/eval/robocasa/groot_n15/steer_eval/$c/final_chain.log)"; done
# 죽었을 때 재발사 절차 (순서 중요):
pgrep -f master_final            # ① 기존 마스터 완전 부재 확인 (있는데 또 발사 = 중복실행 사고)
rm -f outputs/eval/robocasa/groot_n15/FINAL2_ALL_DONE   # ② stale sentinel 제거 (없으면 ab_sweep 체인 오발사!)
setsid nohup bash scripts/safe/groot_n15/robocasa/steer/master_final_scene2.sh < /dev/null > /dev/null 2>&1 & disown
# 집계(수동 아무때나): BREAD_ONLY=1 ~/miniconda3/envs/libero_bench/bin/python scripts/safe/groot_n15/robocasa/steer/aggregate_final_scene.py
```
- 집계 산출: `steer_eval/RESULTS_final_scene_bread.json`(bread 중간) / `RESULTS_final_scene.json`(최종 8-cell) — Notion append 포함.
- ⚠️ **이전 세션의 waiter들은 세션 종료와 함께 소멸** — 새 세션에서 완료 감시를 다시 걸 것(PID 기반, 함정 §2.4-3).
  완료 후 판정: baseline·re-roll null 대비 + per-seed flip 분해.

### 2.4 ⚠️ 함정 목록 (이 세션에서 실제로 당한 것들 — 반드시 숙지)
1. **bash 예약변수 `GROUPS`** — 대입이 조용히 무시됨(gid로 확장). 스크립트 변수명 충돌 금지(FIT_GROUPS로 개명돼 있음).
2. **pkill/pgrep 자기매칭** — 명령줄에 대상 스크립트명 리터럴이 있으면 자기 셸을 죽임(exit 144).
   패턴을 변수로 쪼개거나(`P='foo'"_bar"`) **PID로** 죽일 것.
3. **setsid PID 재부모화** — `$!`는 중간 PID; 실제 프로세스는 pgrep으로 재확인 후 waiter에 사용.
4. **stale sentinel** — 재발사 전 이전 `*_DONE` 파일 삭제. waiter도 sentinel+로그 둘 다 최신 실행 구간만 검사
   (`awk '/START/{n=NR}END{print n}'`로 마지막 START 이후만 tail).
5. **append 로그 오탐** — FIT FAILED 등 grep은 최신 START 이후 구간만.
6. **실행 중 스크립트 파일 수정 금지** — bash가 이어읽다 어긋남. 반드시 kill → 수정 → 재발사.
7. **심링크는 상대경로** — 호스트 절대경로는 컨테이너(/temporal_vla)에서 깨져 serve가 TIMEOUT.
8. **NPZ 로더 α** — §1.1. 명시 필요 시 `STEER_ALPHA` env.
9. **fit 무결성 게이트** — fit은 `.fitlog`로 전체 로그 남기고 `[done]` grep으로 검증(마스터에 내장).
10. CPU: 동료 공유 — serve 총 6~8개 상한에서 load 관찰. 로컬 50% 상한 지향.

---

## 3. 예약된 후속 (자동 체인)

- **α×β sweep**: `scripts/safe/groot_n15/robocasa/steer/ab_sweep_chain.sh` 가 `/tmp/ab_sweep_chain.sh` 실행분으로
  **FINAL2_ALL_DONE 대기 중**(호스트 프로세스). bread84 단일 scene, global **perm**, α{0.3,1(선택값),3}×β{0.1,0.3,0.5},
  ep60–119, GPU4 lane. α=3 재료 fit 포함. 재부팅 등으로 죽었으면 repo 사본으로 재발사.
  ⚠️ SUF 생성식 `a${1/./}b${2/./}` — α=0.3→`a03b01` 형태. 집계는 스크립트에 없음(완료 후 수동 집계).

---

## 4. A100 서버 (worker2 = junhyeong@166.104.35.50, ssh key 등록됨)

- **사양**: 64코어(Xeon 6326×2) load~5, RAM 1TB, **A100 80GB×4 (2,3번 유휴; 0,1번 타인 사용 중 — 불가침)**,
  docker 있음. ⚠️ **디스크 48GB 여유(99%)** — junhyeong에게 정리 협의 필요(그쪽 repo `outputs/`70G+`data/`38G+ckpt 12G가 후보).
- **이미 있는 것**: repo `~/pkt_ws/temporal_vla`(dev@5/13, 낡음), **robocasa Docker 이미지 28.2GB 빌드됨(최대 난관 해소)**,
  GR00T-N1.5 **base** ckpt(HF 캐시; 우리 ckpt-120000은 없음 — 아래 관문 1).
- **준비 완료(2026-07-06 에이전트)**: serve env = conda **`lerobot_050_groot`**(lerobot 0.5.1=로컬 pin 일치, fastapi/uvicorn 설치됨,
  GPU2 CUDA smoke OK). `/cache` 레이아웃 생성(GR00T-N1.5 base 심링크). untracked `docker-compose.override.yml`이
  lerobot→GPU2 pin(유지 — GPU0,1 회피에 유리). lerobot 이미지는 미빌드라 **serve는 conda로 호스트 실행**이 현실적.
- **남은 관문 3개**:
  1. **ckpt-120000 부재(블로커)** — worker2엔 없음. source HF repo(`robocasa/robocasa365_checkpoints`)는 public·rev 일치
     → 첫 serve 기동 시 자동 다운로드 **7.6GB**(권장) 또는 로컬 blob rsync. 승인 필요.
  2. **코드 동기화**: 미커밋 ~56파일(§5) commit+push(승인 필요) → worker2 pull. compose의 `/cache` 마운트·HF_HOME도
     이걸로 해결. 단 **TORCH_CUDA_ARCH_LIST=8.6→8.0**(A100) 조정 필요.
  3. sync 후 스모크 1판 → lane 개설.

---

## 5. 미커밋 상태 (유실 위험 — 우선 처리 권장)

브랜치 `exp/rung2-n15-phase-separation`, 미커밋 ~56파일. 핵심:
- 라벨러(proximity+wrong-grasp) + 테스트 27 pass: `groot_n16/{libero,robocasa}` labeler, `tests/test_*_labeler.py`
- serve 배선: `scripts/serve/lerobot.py`(--steering-layers/-npz-dir/-phase-npz-base + `/steering_phase` 엔드포인트),
  `steering_hooks.py` 무변경, `tests/test_serve_lerobot.py`(3 pass)
- collector: `http_feature_collect.py`(--proximity-phases/--gated-steering/wrong_grasp 필드)
- fit: `steer/fit_phase_conceptor_n15.py`(per-cell/per-phase, carve 옵션)
- 파이프라인/런처: `steer/{master_final_scene2.sh, master_6p_pipeline.sh, heldout_round_cell.sh(범용 N-worker/ARMS/SUF/
  NPZ_ROOT/STEER_LAYERS/STEER_ALPHA/STEER_BETA/PROX), aggregate_final_scene.py, ab_sweep_chain.sh}` 등
- 분석/vis: `analyze/{phase_separation.py, dit_succfail_investigation.py, phase_separation_combined.py}`,
  `vis/{annotate_phase_video.py(6/7-phase 색), phase_distribution_tsne_lda.py}`
- 커밋 시 commitor 에이전트(한글 메시지·논리 단위 분리). PR은 사용자가 원격 agent로([[pr-via-remote-agent]]).
- ⚠️ Isaac-GR00T 서브모듈 내 수정 1건: `gr00t/eval/sim/wrapper/video_recording_wrapper.py`
  (자막을 하단 덮어쓰기→상단 여백; 서브모듈 repo에 별도 커밋 필요).

---

## 6. 기록 위치 (보고/문서)

- **Notion 결과 페이지(seq-steer)**: id `38e63918d42a80698ac2f193716c03a3` — 라운드별 결과 테이블 누적.
  헬퍼 `scripts/utils/notion_seq_steer.py append "한 줄"` (토큰 = repo `.env`의 NOTION_TOKEN).
- **Notion 주간 페이지**: id `39263918d42a8042840cc29325082132` → "1째주" 블록(`39263918-d42a-80af-bef3-ce9ee72c3453`)
  아래 Definite Test 섹션(callout→설계(+연산자 상세 토글: 영문 그림 2장)→결과(9열 테이블)→다음).
  스타일: **notion-polish skill**(user-level) 준수. 이미지 원본 `outputs/eval/robocasa/groot_n15/notion_figs/`.
- 예시 영상 48개(조건×성패): `outputs/eval/robocasa/groot_n15/annotated_6p_samples/` (+ Notion 토글에 16개 업로드됨).
- ⚠️ 구식 문서 주의: `experiment_flow_diagram_prompt.txt`, `docs/steering/` 일부, 플랜 본문의 Rung 기술은
  **구 라벨 체계(event-anchored 단조) 기준** — 라벨러 현행 정의는 본 문서 §1.2가 단일출처.
- 데이터 정책: 완료 라운드는 원격 HDD(kimseungjun@166.104.146.37:11112, `workspace/temporal_vla/outputs/...`) rsync 후
  로컬 대용량(pkl/mp4/csv) 삭제, 요약물(tsv/json/NPZ/문서) 유지. **rsync 검증(rc=0+용량 대조) 전 삭제 금지.**

---

## 7. 다음 로드맵 (사용자 확정 방향)

1. **진행 중 매트릭스 마무리** (§2) → bread 중간 보고 → 전체 8-cell 최종 테이블(Notion) → 판정
   (scene-seed 재현성 / instruction-내 cross-seed / grand 순서로 "일관 방향 존재?" 질문에 답).
2. **A100(worker2) 활용 개시** (§4): 커밋·push 승인 → 코드 pull → 스모크 → lane 개설.
3. **instruction별 scene 수 증가**: manifest(`coast4_reused_remote/manifests/selected_instruction_seeds.tsv`,
   instruction당 50 seed)에서 추가 seed — A100에서 수집/평가 분담.
   **새 cell 추가 절차** = master의 `CELLS` 배열에 행 추가(`cell_id|task|env|task_idx|seed|instruction`) +
   lane 배정 — 수집(S1)·fit(S2)·arm(S3)은 전부 그 배열에서 파생되므로 그 외 수정 불필요.
4. **단일 scene 정밀 sweep**: 예약된 α×β 체인(§3) + 필요시 gated로 확장.
5. **SAE 단계 진입**(그 다음): 설계 완성본 `docs/references/reading_notes/SAE_synthesis_and_design.md`
   (S1 TopK SAE → S2 confound feature 제거 → S3 scene-free conceptor + residual-preserving edit).
   데이터 스케일업(scene 확장)이 SAE 학습 데이터 확보와 겸용. [[sae-study-synthesis]]

---

## 8. 표준·정책 리마인더

per-timestep 유지(rollout-pool 금지 [[feedback-no-rollout-pooling]]) / steer eval은 fit-seed 분리 held-out /
GPU 사용 후 정리·pre-flight [[cleanup-policy]] / 장시간 run은 setsid detached [[long-run-detach-setsid]] /
한글 산출물 [[feedback-korean-docs]], 수식 $ 금지 [[no-latex-dollar]] / 커밋·push는 사용자 승인 후.
