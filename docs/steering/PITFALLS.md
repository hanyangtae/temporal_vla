# 배선·실행 함정

**steering 실험을 돌릴 때 반복해서 사람을 잡은 것들.** 라운드 무관 상시 규약이다.
데이터 취급은 [`../04_data_storage_convention.md`](../04_data_storage_convention.md), scene 필터는 [`SCENE_FEASIBILITY.md`](SCENE_FEASIBILITY.md).

> ⚠ 여기 적힌 것은 **함정과 검증 절차**다. 특정 라운드에서 고른 layer·α·β 값은 그 task·모델·cell
> 한정이라 옮겨 쓰면 안 된다. 하이퍼파라미터는 매번 그 조건에서 다시 고른다.

## 1. α 배선 — 선택값이 조용히 무시된다 (exp2에서 실제 발생)

fit 단계의 α 선택(overlap 밴드)은 **매 fit 실행되고 있었다**. 문제는 serve 쪽이었다.

- 전 fit 1160건의 선택 α 분포: 0.1(49%) / 0.3(28%) / 1(15%) / 3(6%) / 10(3%).
- NPZ에는 `{선택 α, 안전 default 0.3}` 두 키가 저장된다.
- serve는 `STEER_ALPHA` 미지정 시 NPZ **첫 키**를 로드하는데, 저장이 python `set` 순회라
  순서가 hash 우연에 좌우된다. 실측: `{0.1,0.3}` → 첫 키 0.1(선택값 적용됨),
  `{0.3,1}`/`{0.3,3}` → 첫 키 0.3(**선택값 무시, 0.3으로 억눌림**).
- 결과: 실제 적용 α는 `{0.1, 0.3}` 혼합. 선택이 ≥1이었던 **23%는 조용히 0.3**으로 돌았다.

**규약**: ① 선택 α를 fit meta(JSON)에서 읽어 serve에 **명시 전달**(`STEER_ALPHA`)
② NPZ 저장을 `set`이 아닌 **명시 순서**로 ③ 적용된 α를 실행 로그에 남길 것.

**현황(2026-07-31 확인)**: ②는 봉합됐다(커밋 `a6311c5`, NPZ 키 순서 고정) — 같은 NPZ면 항상
같은 키가 뽑히므로 **재현성은 확보**됐다. 다만 `load_steering_matrix`의 폴백
`chosen = steer_keys[0]`(`steering_hooks.py` 81~82행)은 **그대로 남아 있다**. 즉 α를 명시하지
않으면 여전히 **이름이 아니라 파일 내 위치로** 행렬이 정해진다.

- 실측 사례: Per-Step NPZ(`step0..step3_alpha0.3_C_steer`)에 `--steering-alpha` 없이
  `--steering-denoise global`로 걸었더니 **`step0`이 뽑혀 4 denoise step 전부에 적용**됐다.
  의도한 선택이 아니라 첫 키였을 뿐이다. 참고로 step 간 행렬 차이는 작지 않다
  (`‖C_k − C_0‖/‖C_0‖` = step1 0.17 / step2 0.36 / step3 0.54).
- α를 명시하면 이름으로 지정되고, 없는 α면 `KeyError`로 죽는다(75~77행) — **명시가 안전**.

## 2. serve 배선

- `STEER_ALPHA` · `STEER_LAYERS`는 **항상 명시**한다. 기본값에 기대지 말 것.
- `setpoint_seg` 연산자는 `--steering-token-select all`이 **필수**다. 누락 시 기동 실패
  (exp5-3에서 발견).
- `--steering-layers`(lerobot.py) 배선 자체는 유효하다. 다만 multi-layer 이득 주장은
  in-sample 아티팩트였던 이력이 있다 — held-out으로만 판정할 것.

## 3. fit ↔ eval 분리 (가장 자주 재발)

- **fit에 쓴 episode를 eval에 넣으면 in-sample rescue 아티팩트**가 나온다. multi-layer +0.20이
  이걸로 뒤집혔다(held-out 재측정 −0.067).
- 선택(hyperparameter tuning)용 rollout도 fit·eval 양쪽과 **disjoint**여야 한다.
  exp2 재설계에서 60판을 층화 고정-seed 30/30으로 나눈 것이 그 대응이다.
- 검증은 주장이 아니라 **artifacts 기준**으로: `fit episode set ∩ eval episode set = ∅`을
  실물로 확인하고 그 수치를 보고에 적는다.
- fit에 쓴 episode 목록을 **NPZ meta / manifest에 기록**한다. 사후 감사 불가능하면 판정도 불가능.

## 4. fit 표본

- 클래스별 최소 표본이 없으면 대조가 성립하지 않는다. 실측(`phase_event_6p` fit 트리, 파일명
  집계): **9 cell 중 6개가 대조 fit 불가** — 5개는 실패 0판(SR 1.00), 1개는 성공 0판(SR 0.00).
  "고SR scene은 실패 2~6판"이 아니라 **0판**인 cell이 다수다. scene seed를 고정하면 그 scene의
  난이도가 고정되므로, 표본 균형은 수집 전 seed 선정에서 결정된다.
- **빈 fit 게이트 — 봉합됨(2026-07-31 확인)**. 구 배선에서 빈 fit(`{}` 2바이트)이 `[done]`을
  통과한 이력이 있으나, 현재 `fit_phase_conceptor_n15.py`는 유효 group×layer가 0개면
  `[empty]` 출력 후 **`sys.exit(3)`** 으로 죽는다(커밋 `a6311c5`). 러너는 로그의 `[done]`을
  grep하므로 그대로 실패로 잡힌다.
  - **남는 것 = 부분 실패**: 9 group 중 8개가 skip되고 1개만 성공해도 `summary`가 비지 않아
    `[done]`이 찍힌다. 몇 개가 `[skip ...]`됐는지는 **로그에만** 있고 종료 코드에 안 드러난다.
    fit 결과를 쓰기 전 `[skip` 라인 수를 세어 기대 group 수와 대조할 것.
- "첫 N판 고정" 샘플링은 폐기. 제약 층화 랜덤 + 고정 seed를 쓴다.

## 5. 위약(placebo) 대조

- steering 효과 주장에는 **라벨 순열 위약**이 필수다. exp2~exp5 전 라운드에서 위약이 처치와
  같거나 그 이상 움직였다.
- 순열은 **episode 단위**로. record 단위 셔플은 길이 편향을 재유입시킨다.
- 위약도 **dose-match**해야 한다(같은 크기의 개입). 크기가 다르면 방향 효과와 용량 효과가 섞인다.
- ⚠ 순열 위약이 완전히 class-blind가 아닐 수 있다(exp5-2에서 VL heldout AUROC 0.82).
  위약은 "무효"가 아니라 "약화된 방향"일 수 있음을 명시할 것.

## 6. 실행·운영

- 멀티시간 로컬 run은 `setsid nohup`으로 띄운다. agent 백그라운드 잡은 harness가 중간에 죽여
  trap cleanup이 **빈 results + 가짜 `[done]`** 을 남긴다. 완료 판정은 **results 행 수**로 한다.
- 실행 중인 스크립트를 수정하지 않는다(신규 파일 원칙). 러너가 스냅샷을 잡는 구조라 조용히 어긋난다.
- queue 러너: `${var:+X=y}` 셸 확장이 layer 인자를 유실시킨 이력 — `env` 경유로 전달.
- 머신 출처를 `MACHINE.txt`에 기록한다. 머신 간 trajectory는 발산하므로 cross-machine 비교는
  각주 필수(효과 크기 < n=60 검출 한계라 허용은 되지만 밝혀야 한다).
- 에피소드당 **fresh 프로세스**. 한 프로세스에서 `gym.make`를 연속 호출하면 두 번째부터 scene이 오염된다.
- CPU cap `OMP/OPENBLAS_NUM_THREADS ≤16`(공유 노드), GPU는 발사 직전 점유 확인.
- **awk falsy-"0"** (2026-09-01, 2회 재발): 오케스트레이터 그룹핑 `ns[key]=ns[key]?…`가 noise "0"을
  거짓으로 봐 n0 판을 무음 탈락시킨다 → `(key in ns)` 판정. **완주 후 매니페스트 대비 커버리지
  감사**(행수 대조)를 표준 절차로.
- **serve 즉사 ≠ 행업**: 기동 실패(NPZ op 불일치 등)여도 러너 부팅 루프가 `SERVE_BOOT_TRIES`까지
  기다려 30분이 사라진다. 먼저 `serve_<port>.log`의 `startup failed`를 본다.
- **캡처 모드 수집의 집계**: grid 좌표 저장만 하면 `raw_rollouts` sidecar가 없어 INCOMPLETE 오판 →
  러너 재시도가 자기 pkl과 덮어쓰기 tripwire 충돌. 이중기록(7c1a11f) 필수, OUT_ROOT 겹침 금지.
- **좌표 열 혼용**: v4 index의 `armsig`는 전 행 'base'(지터는 `jitter_reset_idx`), `cell_si`는
  평탄 셀코드(scene×100+k), per_episode `scene_idx`는 base scene — 매니페스트 조인 키 확인.
- **수집 라벨은 replay와 다른 세계**일 수 있다(반전 ~59%, GPU·머신·캡처 무관) — 라벨 정본은
  eval 파이프 replay로 재수집한 것(v4r). drawer는 지터 k가 좌/우 방향을 재추첨한다.
- 컨테이너 장기 가동 시 NVML 상실(`Failed to initialize NVML`) → serve CPU 부팅 위장사망 →
  `docker restart lerobot`. pkill 자기매치(`[k]` 브래킷·kill과 발사 분리)·ssh setsid 붙들림(`ssh -f`).

## 7. 통계

- n=60이면 SE ≈ 6.5%p. **조건당 20판이면 MDE +0.43** — null이 "효과 없음"이 아니라
  "검출 불가"일 수 있다. null을 보고할 때 검정력을 함께 적을 것.
- paired 비교가 가능하면 paired로(McNemar). 단 불일치 쌍이 부족하면 검정력이 안 나온다.
- 다중 가설이면 보정(Holm). 탐색 arm은 "미보정·탐색"이라고 명시하고 단정하지 않는다.

## 8. 배선 지점 (코드 앵커)

구 `19_exp3_execution_handoff`에서 file:line 확인된 것. **exp3 시점(2026-07-14) 기준이라
현행 여부는 쓰기 전에 확인할 것.**

| 항목 | 위치 |
|---|---|
| DiT capture (block residual) | `scripts/serve/safe_hooks.py` `assemble_blocks` — hook은 `transformer_blocks[i]` 출력 |
| VL capture (`vlln` 출력 = LayerNorm 후·VL-SA 전) | `scripts/serve/safe_hooks.py` |
| `token_select="all"` 전달 | `scripts/serve/steering_hooks.py` ← `scripts/serve/lerobot.py` (gated/multi/single 3곳) |
| conceptor fit `--denoise {pool,stack,step0}` | `fit_phase_conceptor_n15.py` (α 저장 로직 포함 — §1 참조) |
| task 이벤트 라벨러 등록 | `scripts/safe/groot_n16/robocasa/collect/robocasa_event_labeler.py` `TASK_EVENTS` — **미등록 task는 `http_feature_collect.py`에서 KeyError**. 새 task 수집 전 반드시 등록 |
| drawer 성공 술어 | `src/benchmarks/robocasa/.../kitchen_drawer.py` `_check_success` (joint_p≥0.95 = 성공 → settle phase 관측 불가) |

## 출처

exp2 재설계 문서(구 `17_steering_experiment_redesign.md`, 2026-07-09)의 α 감사·배선 체크리스트를
흡수하고, exp3~exp5에서 추가로 발견된 함정을 합쳤다. 라운드별 결과는 [`RESULTS.md`](RESULTS.md).
