# Activation Patching 인과 상한(존재 증명) 실험 핸드오프

작성: 2026-07-16 (배경 세션). 이 문서는 새 세션이 이 실험을 독립적으로 착수할 수 있도록
목적·설계·선행 사실·구현 앵커·게이트를 담는다. 방법 단일 출처는 여전히
`docs/steering/14_pathway_phase_online_steering.md`이며, 이 실험은 그 하위가 아니라
**그 전제를 검증하는 상위 진단**이다. pq3와 병행 시 충돌 회피 규칙은 §6.

## 0. 목적 — 왜 이 실험이 conceptor 변형보다 먼저인가

pq2 재설계 라운드는 위약 대조 포함 null로 종결됐고(`docs/collab/2026-07-10-steering-redesign-gate1.md`
§최종), COAST positive control(+0.16)은 재현되지 않았다(원인 미상, doc 14 §검증 설계).
지금까지의 모든 실험은 "conceptor를 **어떻게** 밀지"의 변형이었다. 이 실험은 한 단계 위 질문을 측정한다:

> **어떤 activation 수준 개입이든, 이 세팅에서 실패 episode를 성공으로 뒤집을 수 있는가?**

방법: 실패가 확정된 episode를 같은 seed로 재현 실행하되, 특정 시점부터 해당 지점의 hidden을
**같은 cell 성공 episode의 저장 activation으로 통째로 교체(patching)**하고 rollout을 이어본다.
conceptor는 성공 부분공간으로의 *부분 투영*이므로, 저장 성공 activation의 *완전 대입*은
activation 개입이 도달할 수 있는 사실상의 상한(oracle ceiling)이다.

**해석 규약 (사전 등록):**

| 결과 | 해석 | 다음 행동 |
|---|---|---|
| 구제율 ≈ 0 (전 시점·전 창) | activation 개입으로는 실패를 못 뒤집는다 — 실패 원인이 관측/장면/스킬 부재 등 activation 밖에 있음 | steering 계열(SAE 포함) 진입 근거 소멸 → 방향 전환 논의 |
| 구제율 > 0 | 개입 가능한 레짐 존재 확증 + **구제 가능성 곡선**(구제율 vs 개입 시점 t0)으로 유효 개입 윈도우 확보 | conceptor/SAE가 상한 대비 얼마를 잃는지로 문제 국소화 |

부가 산출물: t0 스윕이 주는 구제 가능성 곡선은 "예방적 steering이 가능한 시간 윈도우가
존재하는가"(doc 14의 online 식별 문제와 별개의, 더 근본적인 타당성)에 대한 첫 실측이다.

**공정성 규약**: 이 실험은 존재 증명이므로 oracle 정보(시뮬레이터 phase, 사후 실패 지식,
donor 성공 episode)를 **의도적으로 허용**한다. 보고 시 claim 등급은 반드시
"intervention effect — **oracle-assisted upper bound**"로 한정한다(agent_spec §6,
confound-audit 스킬 §출력 계약). 이것은 배포 가능한 방법이 아니라 접근의 타당성 판정이다.

## 1. 선행 사실 (이 세션에서 레포 실측 확인, 2026-07-16)

1. **재현성 전제는 이미 확증됨**: pq3 Phase B **S4 natural-reset 재현성** — 동일 seed·noise
   → 전 필드 동일 (`docs/steering/20_pq3_phase_c_handoff.md` §2). 따라서 실패 episode를 같은
   (env_seed, noise_seed)로 재실행하면 baseline이 결정적으로 재현되고, 패칭 효과를
   **paired**로 측정할 수 있다. 별도 결정론 게이트는 스모크 수준(n=3)이면 충분.
2. **주입 인프라 존재**: `scripts/serve/steering_hooks.py`의 `ConceptorSteering`이
   DiT residual(`action_head.model.transformer_blocks[i]`, D=1536) / DiT 최종(`action_head.model`,
   D=1024, pre-velocity) / VL(`action_head.vlln`, D=2048)에 forward hook을 이미 건다.
   현재는 `h' = h·Mᵀ` 변환만 지원 — **"저장 시퀀스로 교체" 모드가 신규 구현 대상**(§4).
3. **Donor 데이터 가용성** (`docs/steering/NOTICE_pq2_fit_loss_for_pq3.txt` 필독):
   - 생존: 구 3-scene cell(bread/potato/apple) fit 원료 **180판, 승준 datasets에 실물 확인됨**.
   - 유실: pq2 seed-변형 5 cell fit 원료. 필요 시 (scenario_seed, inference_seed) 결정적
     재수집 가능하나 **보류 결정(사용자)** — 이 실험은 생존 cell 또는 pq3 신규 수집분으로 시작.
   - pq3 수집분 pkl에는 env-step GT(`env_step_phases/env_step_success`)가 자동 포함(커밋 ea61d24).
4. **용량 감각**: full-token 캡처는 timeout 실패 1.09GB/ep지만, donor는 **성공 episode(짧음)**만
   필요하고 표준 7층 record 캡처(CAP 0,2,4,8,10,12,15)는 훨씬 작다. patching rollout 자체는
   **캡처 OFF**(`--no-features`)로 돌린다 — 주입만 하고 수집하지 않음.
5. **서버 호칭**: 승준 = kimseungjun@166.104.146.37:11112 / a100 50 / a100 48 / 로컬
   (w1/w2 호칭 금지, doc 20 §0). SR rollout은 **로컬 전용**, 원격은 분석·추출까지만(CLAUDE.md).

## 2. 실험 설계 v1

### 대상과 donor

- **대상 실패 집합**: 선택 cell의 baseline per-episode TSV에서 실패 판정 episode
  (seed·noise 시리즈 기록 있음). cell 선정은 사용자 게이트 — 권장 기준: base SR 0.3~0.7
  (천장/바닥 회피; 참고: drawer_left tail 10판 base 3/10, doc 20 §2). 천장 cell(apple류)은
  구제할 실패가 적고, 바닥 cell은 실패 원인이 activation 밖일 가능성이 높다.
- **Donor**: 같은 cell 성공 episode의 저장 activation 시퀀스(승준 생존 180판 또는 pq3 수집분).
  매칭 v1 = **phase 시작점 정렬**: 개입 시점 t0가 속한 oracle phase의 시작 record부터
  donor의 같은 phase 시작 record를 맞춰 순서 재생. (시간 워핑·최근접 이웃 매칭은 v2 —
  v1 신호가 있을 때만.)

### 개입 축 (작게 시작)

| 축 | v1 값 | 근거 |
|---|---|---|
| 개입 시점 t0 | phase 경계 기준 3지점: reach 초입 / grasp(or pull) 진입 / 실패 발산 직전 | 구제 가능성 곡선의 최소 해상도. 발산점은 사후 라벨(env-step GT·mp4)로 정의하고 문서에 정의를 동결 |
| 창 길이 W | ① 1~3 record(5~15 env-step) 후 자유 진행 ② episode 끝까지 지속 | 두 극단만. 지속 패칭은 사실상 open-loop activation 재생임을 인지(아래 주의) |
| 주입 지점 | Stage1 선택 layer 1개 + DiT 최종(D=1024) 2개만 | 폭발 방지. VL 축은 v1 제외 |
| denoise K | donor 저장이 K별로 있으므로 K-정렬 재생 (K=4) | record 구조 [L,K,T,D] 그대로 |

### 대조군 (필수 3종)

1. **no-patch 재실행**: 같은 seed·noise, hook 미등록 — 결정론 확인 겸 paired base.
2. **placebo-fail**: 다른 **실패** episode의 activation을 동일 방식 주입 — "아무 교란이나
   결과를 흔드는가"의 방향성 통제 (pq2에서 위약이 최대 양성과 동급이었던 교훈).
3. **donor-shuffle**: 다른 cell/scene의 성공 donor 주입 — "성공 정보"인지 "임의 강한 신호"인지 구분.

### 규모와 판정

- 시작 규모: cell 1~2개 × 실패 10~15 ep × (t0 3 × W 2 × arm 3 + base 1) ≈ 200~570 rollout.
  캡처 OFF라 rollout당 비용은 eval과 동일. 신호가 보이면 확대, 전무하면 그 자체가 결론.
- 판정: 실패→성공 **구제율**, 같은 seed paired 비교, exact McNemar (pq2/pq3 판정 도구 재사용
  가능: `scripts/safe/groot_n15/robocasa/steer/pq3/pq3_decision.py`의 검정 부분 참조).
- 성공 판정은 corrected 기준(apple류 cell이면 rejudge 규약, `docs/steering/18` §재판정).

### 주의 (설계 한계, 보고서에 명시할 것)

- **Off-policy 발산**: 패칭 직후 실제 관측은 donor의 관측과 다르므로, 지속 패칭(W=끝까지)은
  "성공 activation의 open-loop 재생"이 된다. 그래서 W 두 극단을 모두 측정한다 — 짧은 W는
  "복귀 유도" 가설을, 지속 W는 "activation이 행동을 지배하는가"의 상한을 각각 검증.
- **관측-activation 불일치**: 패칭 중 모델의 다른 경로(예: VL, 패칭 안 한 layer)는 실제 관측을
  계속 본다. 이 불일치 자체가 상한을 낮출 수 있음 — 결과 해석 시 "layer 국소 패칭의 상한"임을 한정.
- 구제율이 t0=발산 직전에서만 0이 아니라면: 예방 윈도우는 좁고 감지기 결합이 필수라는 뜻.

## 3. 실행 절차 (요약)

1. **계획 반론 게이트**: 착수 전 codex-collab 규약대로 Codex 반론 게이트 1회
   (`scripts/utils/codex_ask.sh` 경유, `.claude/skills/codex-collab/SKILL.md`). 이 문서가 계획 초안.
2. **cell·데이터 확정** (사용자 게이트): 대상 cell, donor 출처(승준 생존분 vs pq3 수집분),
   GPU/시점(§6).
3. **donor NPZ 추출** (승준, 원격 규약): 생존 fit 원료 pkl에서 (episode, layer, K, record 시퀀스)
   추출 → 소용량 NPZ만 회수. `scripts/utils/remote_compute.sh`, python은
   `~/anaconda3/bin/python`(base python3에 torch 없음, **scipy는 어디에도 없음**).
4. **patch hook 구현 + 유닛** (§4): pq3와 파일 분리(신규 파일), 브랜치 `exp/patching-ceiling`
   (pq3 브랜치 `exp/pq3-coast-align` 수정 금지).
5. **스모크**: 결정론 재확인 n=3 (no-patch 재실행 = 원 결과 일치) → sham patch(donor=자기 자신)
   = base와 완전 일치 확인 (pq3 S2b sham 패턴 재사용).
6. **본 실행** (로컬, 캡처 OFF) → 판정 → confound-audit 표 첨부 보고(§5).

## 4. 구현 앵커

- `scripts/serve/steering_hooks.py`: `ConceptorSteering.__init__/register`가 hook 배선의 기준.
  신규 `PatchSteering` 클래스(또는 mode 분기): hook에서 M 곱 대신
  `output[..., token_sel, :] = h_donor[k_step, t_cursor]` 로 교체. record 진행 커서는
  pq3 Per-Step 훅의 `reset_step_counter()`/step 카운트 패턴 재사용(요청마다 K회 발화).
- `scripts/serve/lerobot.py`: 기존 `--steering-*` 플래그와 같은 결로
  `--patch-npz / --patch-layers / --patch-start-record / --patch-len` 추가. 시작 시점은
  serve 쪽 record 카운터로 판정(클라이언트 변경 불요)하거나, `/steering_phase`처럼 HTTP
  트리거를 추가해도 됨 — 구현 세션 판단.
- token 선택: v1은 `token_select=all`(pq3 표준, 49-token 정렬). donor 저장 형식과 일치 필수.
- 검증 표준: 삭제·이동 전 보존 검증은 NOTICE §3 (이름 세기 금지, `find -type f` + `du` +
  평균 파일크기 상식 체크).

## 5. 게이트·보고 규약

- 보고 전 **confound-audit 스킬 필수**. 이 실험은 oracle-허용 존재 증명이라 gate 다수가
  N/A지만, **표를 채우고 N/A 사유를 명시**한다 (특히 gate 4: fit/eval 분리는 donor≠대상
  episode로 대체 충족, gate 8: cell 1~2개 결과는 "해당 cell 조건부"로 한정).
- claim 등급: "intervention effect — oracle-assisted upper bound". SR 개선 방법 주장 금지.
- 산출물: ① cell×arm×t0×W 구제율 표(n 명시) ② 구제 가능성 곡선(구제율 vs t0) ③ 판정 문구
  (성공 기준: placebo-fail·donor-shuffle 대비 donor-success가 유의하게 높은 구제율).
- 보고는 한국어, 문서는 `docs/steering/` 다음 번호로.

## 6. pq3 병행 충돌 회피 (중요)

- **실행 중 스크립트 수정 금지** (HANDOFF §8 실증 사고: bash 재읽기 정렬 사고). 신규 파일로만.
- 로컬 GPU는 pq3 eval이, 승준은 pq3 fit·아카이브가 점유 — **실행 시점과 GPU 배정은 사용자
  게이트**. pq3 유휴 구간(수집 완료~판정 대기)에 끼워 넣는 것을 권장.
- 디스크: patching rollout은 캡처 OFF라 mp4·TSV 수준. donor NPZ는 소용량만 로컬 회수.
  승준 HDD 여유는 pq3 full-token 아카이브(~80GB 예정, doc 20 §2)가 우선.
- 컨테이너 재시작 금지(수집기·VNC 세션 끊김 사고, NOTICE §2). a100 48은 robocasa 컨테이너
  GPU 상실 상태였음(doc 20 §2) — 쓰려면 사용자와 재시작 협의.

## 7. 미결 사항 (착수 세션이 사용자에게 확인)

1. 대상 cell 1~2개 (권장: base SR 중간대 + donor 실물이 있는 cell 교집합).
2. donor 출처: 승준 생존 180판(bread/potato/apple) vs pq3 신규 수집분(drawer/ppcc 신규).
3. 실행 시점·GPU 배정 (pq3 일정과 조율).
4. t0 "발산 직전" 정의 방식 (env-step GT 기반 자동 vs mp4 수동 라벨) — 동결 후 진행.
