# 47 — Per-step 게이팅 파이프 (설계 정본)

2026-08-26 사용자 확정. **latch 설계(발화 후 에피소드 끝까지 상시 적용)는 의도에 없던
것으로 전면 폐기** — latch 전제의 파이프/연산자 라운드 문서(구 42·44)와 관련 핸드오프는
삭제했다(내용은 git 이력에만 존재). latch arm으로 측정된 개입 판정은 이 설계를 구속하지
않는다.

## 1. 설계 원칙

1. **per-step 게이트**: 매 inference step(get_action, 5 env-step 해상도)마다 게이트가
   (개입 여부, 어떤 연산자)를 결정한다. 개입은 **그 step 1회성** — 다음 step은 다시
   무개입이 기본값이고, 게이트가 다시 발화해야 다시 개입한다.
2. **detector 입력 = 항상 pre-hook 활성화** (steering hook 적용 **전** 값, 발화 전후
   불문·모든 step). 근거 = 순환(Goodhart) 차단: 연산자는 detector가 읽는 같은 공간
   (DiT L12/15)을 성공 쪽으로 미는 것이라, steer된(post-hook) 값을 먹이면 점수 하락이
   개입의 **정의**가 되어 게이트가 자가 소등한다 — latch 파일럿에서 실측(oven rs_setm:
   개입 후 p 0.010 ≪ δ 0.177인데 실제 실패). pre-hook 스트림은 "steer된 행동이 실행된
   세계에서 모델이 스스로 만든 상태"라서 개입의 인과 효과는 세계를 거쳐 다음 관측으로
   들어온다. LSTM hidden 롤백/재갱신은 필요 없다(단일 pre-hook 스트림으로 자기일관).
3. **재발화가 재시도를 포섭**: 개입 다음 step의 자연 활성화가 여전히 실패 쪽이면
   게이트가 재발화 → 재개입(같은 연산자 반복 또는 교체). K-재시도를 별도 장치로 만들지
   않는다. 재발화 상한·연산자 교체 규칙은 미결 설계 질문(§4).
4. **판정 종점은 env SR**. detector 점수는 게이트 신호일 뿐 개입 효과의 지표로 절대
   쓰지 않는다(위 순환 때문에 원리적으로 불가).

## 2. 부품 좌표 (유지 — latch와 무관하게 검증된 것)

- **detector**: per-task SAFE-LSTM, **phase-gt 길이 절제** 학습(43 — 성공 판 phase별
  dwell ceil(μ+1σ) cap, TRAIN·CALIB만 절제·TEST full). ckpt =
  `.claude/worktrees/safe-length-ablation/outputs/analysis/grid_phase/detector_trunc/phase-gt/detector_pertask_lstm_<slug>.pt`.
  입력 좌표 = DiT L12 · 마지막 denoise · 49토큰 mean [1536]. functional-CP δ, α=0.1.
  런타임 `src/failure_online/online_failure.py` (1-step ≡ 배치 forward 수치 일치).
- **phase**: GT event labeler (client가 매 step `/steering_phase` POST — 개입이 참조하는
  값과 동일).
- **연산자 인벤토리**: setM(`scripts/fit/fit_setm.py` seg 포맷) · condg
  (`scripts/steer/online_gated/fit_cond_guidance.py` — per-scene centering·margin 게이트,
  `--episode-manifest` 경로 기반 fit) · reseed(denoise noise 재추첨 — 성질상 원래
  1회성이라 per-step 게이트와 자연 정합). COAST global conceptor는 이 데이터에서
  활성화 소거기로 판정(6/8 케이스 전 α 퇴화) — 후보 제외.
- **fit 데이터 규약**: v4 지터 격자 케이스별 매니페스트
  (`scripts/steer/online_gated/select_rescue_cases.py --fit-manifest-dir`; 저장 규약
  docs/04, k-grid는 §3.1.1). 구제 케이스 기준: ① scene succ>5 ② 대상 k succ 1~4
  ③ fit = 나머지 k 전부 + 대상 k 실패 ④ 나머지 k에도 실패 ≥1.
- **replay 무대**: v4 지터 격자 — (base env_seed, ep_meta, reset_idx k) bit 결정적,
  replay 결정성은 3머신 40/40 셀 재현으로 검증됨(latch-무관 인프라 사실). detector
  실전 발화도 시뮬 예측과 일치 확인됨.

## 3. 배선 변경 (구현 필요 — 현행 코드는 전부 latch 전제)

1. **pre-hook 캡처 분기**: 현행은 SAFE 캡처 hook이 매 호출 등록되어 steering hook
   (arm 등록 시 선등록) **뒤**에 실행 → detector가 post-hook 값을 받는다
   (`scripts/serve/lerobot.py` `_failure_from_hidden` ← `features.hidden_states`).
   detector 전용으로 steering 적용 전 값을 읽는 캡처를 `safe_hooks`/`steering_hooks`에
   추가한다. fit·분석용 캡처는 post-hook 유지(무엇이 실행됐는지의 기록).
2. **per-step arm/disarm**: `/steering_phase`의 latch 의미(한 번 on이면 계속) 제거 —
   record 단위 on/off + 연산자 지정. `--steer-from-record` latch 옵션 대체.
3. **serve 게이트 루프**(라이브 모드): 매 `/act`에서 pre-hook 점수 → 발화 시 그
   record만 steer. 발화 시 해당 record forward를 steer로 재실행할지(action 교체)
   다음 record부터 적용할지는 구현 시 확정(재실행이면 추론 1회 추가 — 허용 범위).
4. **러너**: trigger 사전표(base pass → make_triggers) 방식은 라이브 게이트 arm으로
   대체. 사전표 방식은 결정성 디버그 용도로만 잔존.

## 4. 미결 설계 질문

- **연산자 family 선택 신호**: phase별 선택은 기존 phase-follow 구조로 해결되지만,
  family(setM vs condg vs reseed)를 step마다 무엇으로 고를지는 미정.
- **재발화 상한·교체 규칙**: 같은 연산자 N회 무효 시 교체/포기 기준.
- **FP 억제**: per-step 게이트에서 FP 1회의 비용은 1-step 개입으로 줄어들지만(latch
  대비), 억제 규칙은 여전히 필요.

## 5. 이전 라운드에서 승계하는 판정

승계(latch-무관): 선형 개입 단독 null 전 계열(exp2–5)·resample 5/65(7.7%)·detector
per-task 채택과 phase-gt 절제(43)·k(배치) 축이 성패 지배·COAST global=소거기·
read≠write(위 §1-2의 순환 실측).

**비승계**: latch arm의 개입 판정 전부(setM/condg/coast의 latch 파일럿 구제/실패 표
포함) — 설계가 의도와 달랐으므로 per-step에서의 효과 여부는 열린 질문으로 되돌린다.
