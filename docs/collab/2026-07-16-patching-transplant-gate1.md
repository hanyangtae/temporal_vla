# Gate 1 — Activation Patching "상한" 실험 계획 반론 (Codex)

- 게이트: Gate 1 (계획 토론). thread_id: `019f6910-b493-7040-a11b-d6451dc5edb3` (1라운드).
- 계획 초안: `/home/dongkyu/.claude/plans/pq3-wise-mist.md` (doc 21 기반, pq2 데이터 단독 변형).
- Codex 역할: 비판적 리뷰어 (반례·confound·단순 대안). 아래 채택 항목은 전부
  **레포 코드로 검증 후** 수용 (규약: 무검증 채택 금지).

## 라운드 1 요지

Codex 판정: "현재 형태로는 판정 불가 — 이 실험이 식별하는 것은 특정 donor 시퀀스를 특정
DiT cut에 이식한 효과뿐. '상한'이라 부르거나 null로 steering 전체를 폐기할 수 없다."

### 채택 (검증 완료)

1. **상한 논리 기각 → 재명명**: 완전 대입은 최적 개입의 상한이 아니라 **구성적 하한**
   (R(I_d) ≤ sup R(I)). 강한 개입이 약한 개입을 지배한다는 단조성 없음 (반례: donor
   hidden이 "이미 정렬됨" 가정의 행동을 내보내 target의 4cm 어긋난 gripper에서 실패).
   → 명칭 **"donor-trajectory transplant probe"**, claim 등급
   "intervention effect — specified donor-trajectory transplant, cell-conditional".
   해석표에서 "구제율 0 → steering 방향 전환" **삭제** (null은 이 transplant class의
   null일 뿐).
2. **L15 포함 all7 arm은 L15 arm과 기능 중복**: DiT block loop 후 norm_out+proj_out만
   있고 residual bypass 없음 — 검증 `cross_attention_dit.py:300-306`. L15 출력 교체는
   그 forward의 upstream 전부(L0-L14 패치 포함)를 지움.
   → arm 재구성: **L15(최종 block) vs early/mid(L15 제외)** 분해.
3. **anchor ≠ donor action 재생**: get_action은 `torch.randn`(target inference_seed)
   초기 noise에서 K회 velocity 적분 — 검증 `flow_matching_action_head.py:361-`.
   L15 전 K 패치 시 action = ε_target + Δt·Σv_donor ≠ donor action(= ε_donor + …).
   → anchor 3단 분리: ① 환경 anchor(donor action open-loop replay = donor 성공 재현)
   ② **action-equivalence anchor**(inference_seed까지 donor로 → emitted == donor 저장
   action 수치 일치; 배선 유닛테스트) ③ 본 개입(target seed 유지, 구제율에 사전 기대 없음).
4. **donor 고갈·phase 정렬 규칙 사전 등록**: t0는 baseline rollout에서 **사전 고정한
   record index**(patched run의 phase 트리거 금지 — treatment가 treatment 이후 상태에
   의존). env-step→record floor/ceil 규칙·phase occurrence index(first_grasp 등) 명시.
   phase 미도달 ep는 제외가 아니라 **ITT(비구제)** + eligible-조건부 병기. 지속 arm은
   donor 끝에서 **즉시 채점**("donor horizon 내 성공") — freeze/loop/복귀 금지.
5. **검정력**: n=12~15는 부족 (control 0·rescue 0.2에서 power 16%/7%). → **실패 전량**
   (s300033 35 + s400020 39 = 74판), **사전등록 primary contrast 1개**
   (L15 × early-t0 × donor-horizon), primary statistic = max(p_placebo, p_shuffle) 단일 α,
   나머지 t0/W 곡선은 hierarchical(primary 통과 후 exploratory). n=35/39 power ≈ 86/91%
   (control 0 가정).
6. **donor가 실험 단위**: donor 1개 재사용은 "한 donor trajectory의 n회 적용".
   → cell당 donor 3~5개 균형 배정, control에도 donor identity 유지, donor별 결과 +
   leave-one-donor-out 보고.
7. **serve cursor 규약**: 기존 `reset_step_counter()`는 요청 내 denoise K만 리셋
   (`steering_hooks.py:271`). rollout record cursor는 별도 상태 → `/reset` 연동 +
   rollout마다 (donor id, t0, W) 원자적 arm API + `/health` fingerprint 필수.
8. **대조 arm 추가**: direct donor-action replay arm(양성이 activation-국소화인지
   action 재생인지 구분) + same-cell donor의 time/phase-shuffle arm.
9. **일반화 한정**: 결론은 "ppcc_bread_s300033/s400020 조건부 feasibility"로 고정.
   양성이어도 방향 유지 결론은 다른 object·task replication 후.

### 비적용/수정 채택 (우리 상황과 불일치 — 근거 명시)

- **"재수집 회피" probe 순서** (저장 action replay 먼저 → pooled mean-shift smoke →
  최소 재수집): 이 두 cell은 **full action이 유실**됨 — 생존 csv는 chunk 첫 action만
  저장하는 스템(`collect_schema.py:30-45`, `first_env_first_step`), pkl은 2026-07-14
  purge. Codex도 "raw trajectory 없으면 복원 불가, 일부 재수집 필요" 인정. →
  **pass A 경량 재수집(60판×2, 결정적)은 불가피**하며, 이것이 환경 anchor·발산점
  라벨·action-replay arm 재료를 겸함. pooled mean-shift smoke는 생존 pooled donor가
  이 두 cell에 없어 생략(다른 cell로의 smoke는 대상 불일치로 가치 낮음).
- **full-token 재수집 축소는 채택**: 성공 donor 전량×7층이 아니라 **donor 3~5판/cell,
  CAP는 arm에 필요한 layer만**(primary L15; secondary용 early/mid 추가 여부는 사용자
  게이트) — 용량 ~1.5GB(1층)~10GB(7층) 수준.

## 이견 (남은 것)

없음 — 치명 지적은 전부 검증 후 채택. Codex의 "570-rollout 본 실행 보류, direct action
replay → action-equivalence anchor → 단일 primary arm 순" 권고를 실행 순서로 수용.

## 추가 정정 — "유실 5-cell" 판정 뒤집힘 (사용자 지적, 2026-07-16 같은 날)

사용자 지적("activation 데이터 승준에 있을 텐데")으로 승준 HDD 를 재검증한 결과:

- **s300033/s400020 fit 수집분(ep0-59) pkl.zst 60판×2 실존**:
  `~/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/phase_event_6p/raw_rollouts/`
  `PickPlaceCounterToCabinet/<cell>/task5--ep*--succ*.pkl.zst` (du 707M/614M).
  NOTICE_pq2_fit_loss_for_pq3 의 "유실 5-cell" 판정은 **`.pkl` 만 세고 `.pkl.zst` 를
  놓친 검증 구멍**으로 추정 — NOTICE §3 이 경고한 "이름 세기" 함정의 변형.
  (NOTICE 본문 정정은 pq3 세션 혼선 방지를 위해 pq3 안정 후 — 사용자 중계 권장.)
- 1판 원격 스키마 실측 (s300033 ep22): `hidden_states` [7,4,1536] fp16 **토큰-풀링본**
  (full-token 아님 — donor 재수집 필요성은 유지), `actions` 90 record × 16-step 전체
  chunk, `feature_phases`/`phase_timeline`/`event_steps`/`grasp_steps` 동봉,
  inference_seed=ep×1000 공식 실측 일치. succ 분포: s300033 20/60, s400020 23/60.

**계획 영향 (v2 → v2.1)**:

1. **targets 재정의**: ho_base ep60-119(actions purge 소실) → **fit ep0-59 실패
   40+37=77판** (actions·phase 라벨 실존 → 발산점 라벨·direct action-replay arm 재료 확보).
2. **pass A 전면 재수집(120 GPU rollout) 삭제**: 발산점 라벨은 pkl 의 record-해상도
   phase/event 필드로, env-step 해상도가 필요하면 retro replay(sim only)로. 결정론
   게이트는 스모크 n=3~5 재실행 + **actions 수치 대조**(csv 시대보다 강한 게이트)로 대체.
3. **GPU 재수집은 pass B 만**: donor 성공 3~5 + placebo-fail 3~5 + sham 대상 소수/cell
   ≈ 15~25 rollout (구 계획 120 대비 ~1/6).

## 사용자 결정 (기록)

- 대상 cell = s300033 + s400020 (scene 2개), layer 축 = 전층+L15 → **Gate 1 후 수정**:
  L15/final vs early/mid 분해로 재구성 (사용자 승인 대기).
- (대기) 재설계 v2 승인 + pass A GPU 시점.
