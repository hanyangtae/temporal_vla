# exp3(구 pq3) 실행 핸드오프 (새 세션용)

작성: 2026-07-14. 계획 세션(설계·Codex Gate 1·데이터 정리)에서 실행 세션으로 인계.

## 0. 필독 — 계획서 단일 출처

**`/home/dongkyu/.claude/plans/dynamic-riding-aurora.md` (v9 승인본)** — exp3 라운드의 설계 전체가 여기 있다.
이 핸드오프는 계획서에 없는 세션 컨텍스트만 담는다. 충돌 시 계획서가 우선.

한 줄 요약: COAST 축 전부 정렬(49토큰 full-token capture→fit-시 mean, **denoise Per-Step**(M_k 스와핑 신규 배선),
layer/α regime별 기하 선택, β는 arm별 fit-seed sweep) + **scene-diverse**(cell=instruction, fit 15 scene,
eval 30 = seen 15/unseen 15) + arm 4개(base/cross_scene perm/gated/null) × 5 cell × 30판.

## 1. 이 세션에서 확정된 배경 상태 (2026-07-14 기준)

- **exp2(구 pq2) P3 라운드 완주·종결**: 115 arm × 60판 전부 완료(ledger), 결론 = 모든 steering arm이 위약 동률(null).
  집계: `outputs/eval/robocasa/groot_n15/steer_eval_exp2/aggregate_v2/` (arms=115).
- **eval activation 전면 삭제됨** (3 호스트, 22,675개/~172GB): steer_eval*·exp2 p2/p3의 rollout pkl/zst는
  로컬·srv50(.50)·승준(.37) 어디에도 없다. 판정은 dual_scoring.tsv 사이드카(115/115) + 파일명 succ 스템(.csv 마커)로 유지
  — **aggregate는 정상 동작(IDENTICAL 검증됨), pkl을 찾으려 하지 말 것**.
- fit activation은 승준(phase_event_6p, 1,605 pkl)에 완비 + 로컬에 apple 서브셋(~2.8G). conceptor NPZ는 로컬 33G + 승준.
- 판정 메타·ledger·mp4·conceptor NPZ는 승준 HDD에 아카이브 완료.

## 2. 파이프라인 규약 (fork 세션 사용자 확정)

- fit 수집: 로컬 GPU **0/1/2** (+필요시 .50) → **수집 즉시 승준 직송** (fit activation은 승준에만 보관)
- conceptor fit: **승준 단일 수행** (anaconda python, 스레드 cap; 분산 fit 기각됨) → NPZ만 전 서버 공유
- eval: 캡처 OFF, 분산(로컬 0/1/2 + .50 + .48) → **판정 데이터만**(사이드카·스템·mp4·ARM_SPEC·ledger) 승준행
- eval activation은 앞으로도 저장하지 않는다 (memory: eval-activation-purged)

## 3. 검증된 코드 앵커 (이 세션에서 file:line 확인 완료)

| 항목 | 위치 |
|---|---|
| DiT capture(last-16 mean → full-token 개조 지점) | `scripts/serve/safe_hooks.py` `assemble_blocks` :254-261 (hook은 transformer_blocks[i] 출력=block residual) |
| VL capture(vlln 출력 = LayerNorm 후·VL-SA 전) | safe_hooks.py:188-195, :305 — post-VL-SA full-token 1지점 추가 필요 |
| token_select="all" 기구현(전달만 안 됨) | `scripts/serve/steering_hooks.py`:161-164, dit 생성 3곳 = `scripts/serve/lerobot.py`:611(gated)/:635(multi)/:664(single) |
| fit --denoise {pool,stack,step0} | `fit_phase_conceptor_n15.py`:186 (Per-Step용 step별 conceptor 산출은 신규), α 저장 로직 :141-142 (--alphas 단독 지정 시 확정 저장) |
| **OpenDrawer 수집 하드 블로커** | `scripts/safe/groot_n16/robocasa/collect/robocasa_event_labeler.py` TASK_EVENTS :163-167 미등록 → `http_feature_collect.py`:488이 무조건 라벨러 생성 = KeyError. **Phase A 최우선** |
| drawer 판정/phase 술어 | `src/benchmarks/robocasa/.../kitchen_drawer.py` `_check_success` :172-192 (joint_p≥0.95=성공 → settle phase 관측 불가; 2-phase placeholder, **외부 세션 산출물 대기**) |
| N1.5 DiT 시퀀스(49=1+32+16 가정) | `lerobot/src/lerobot/policies/groot/action_head/flow_matching_action_head.py`:384 sa_embs=cat(state,future32,action16) — S1 스모크에서 실측 |
| inference-seed(diffusion x₀) 배선 | lerobot.py:231-247 `_apply_inference_seed`(per-call torch.manual_seed), 클라이언트 http_feature_collect.py:186-193 |
| cell 테이블 | `scripts/safe/groot_n15/robocasa/steer/queue/queue_lib.sh` CELLS (pq3_ 5행 추가 필요) |
| seed 소스 manifest | `outputs/eval/robocasa/groot_n15/coast4_reused_remote/manifests/selected_instruction_seeds.tsv` (15 cell×50 canonical seed; drawer L/R=idx8/7, bread=idx5). ppcc 신규 2종은 `select_instruction_seeds.py`로 스캔(potato 제외) |
| exp2 파생 원본들 | `scripts/safe/groot_n15/robocasa/steer/exp2/` (exp2_cell_runner.sh·p0_gate.py·make_fit_manifests.py·aggregate_v2.py·p3_lane_*.sh) |

## 4. 함정 (이 세션에서 실증된 것)

- OpenDrawer는 **natural reset만** — drawer_side가 ep_meta에 없어 `--replay-ep-meta`는 side가 뒤집힘.
- α band 자동선택은 quota 퇴화 α(C_steer≈0 → M≈0.9I 균일수축)를 고를 수 있음 → **quota floor guard 필수**.
- make_fit_manifests.py는 디스크의 전 episode를 층화 split — exp3는 `make_exp3_manifests.py`로 seed 구획 강제.
- succ_ever_th(corrected 채점)는 PPCS(stove) 전용 — drawer/ppcc는 env 원판정.
- 러너 preflight에 token_select·α key·matrix SHA 대조 추가(exp2의 α 오배선·무효 gated 사고 재발 방지).
- gated per-step×phase-bin은 record가 phase dwell ÷ step 4로 얇아짐 — record 하한 게이트 + phase 병합.

## 5. 운영 규칙

- Codex 협업: `.claude/skills/codex-collab/SKILL.md` — 구현 완료 후 **Gate 2 코드 리뷰**(`scripts/utils/codex_ask.sh review`).
  Codex는 gpt-5.6-sol·xhigh로 설정돼 있음(~/.codex/config.toml). Gate 1 원장 커밋 필요:
  `docs/collab/2026-07-13-exp3-gate1.md` (thread 019f5a76-68eb-7d91-9312-2646e78a04cb, R1 10건/R2 6건 요지·수용/기각·사용자 결정 — 계획서 이력 참조).
- 커밋: 한글 prefix(feat:/fix:/docs:), dev에서 분기. gh 없음 — commit+push까지만.
- CPU cap 40%(스레드 ≤16), 승준 HDD만 사용(NVMe 금지), 멀티시간 run은 setsid nohup + results 행수로 완료 판정.
- 보고 전 confound-audit 스킬 필수. Notion/보고서에 $ 수식 금지.

## 6. 실행 순서 (계획서 Phase 그대로)

A 코드 정렬(+Gate A 테스트) → B 스모크 S1-S9(특히 S1 T=49 실측, S2c per-step M 스와핑) → C0 ppcc 신규 2종 seed 스캔 ∥ C fit15 수집(scene 15개·즉시 직송)+게이트+eval seed 동결 → D 승준 fit(Stage1 regime별 layer — **사용자 보고 gate**, Stage2 α, per-step×phase NPZ, β sweep 2종)+hash 동결 → E eval 600판 → F 판정(6-Holm paired McNemar, seen/unseen 분해). fit30 라운드·per_scene 2단계·confirmatory는 각각 별도 재가.

## 7. 대기 항목

- drawer phase 정의: 다른 세션에서 설계 중 — 산출물(코드/문서) 오면 2-phase placeholder 교체.
- Stage1 layer 선택 결과, S1 T 실측, Gate C 4-cell fallback — 각각 사용자 보고/승인 지점.
