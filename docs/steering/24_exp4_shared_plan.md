# exp4 공유 계획 — oracle-timing 구제(exp4-1) + perturbation 유도 실패 conceptor(exp4-2)

작성 2026-07-21, **확정 2026-07-22** (Codex Gate1 1라운드 반영 + 사용자 최종 결정. 원장: `docs/collab/2026-07-22-exp4-plan-gate1.md`). 이 문서는 exp4 두 세션(24a=exp4-1, 24b=exp4-2)이 공유하는 배경·정의·계약·자원 규칙의 단일 출처다. 각 실행 세션은 **자기 문서 + 이 문서**만 읽으면 자기완결이다.

## 0. 명명·확정 스코프

- 과거 라운드 호칭: pq1/2/3 → **exp1/2/3** (디렉토리·브랜치·파일명 등 실물 이름은 그대로, 호칭·신규 문서만).
- 이번 라운드 = **exp4**: **exp4-1**(oracle-timing 실패 구제), **exp4-2**(perturbation 유도 실패 → conceptor). 서로 다른 세션에서 병행 실행.
- **확정 스코프 (2026-07-22 사용자 결정)**:
  - 연산자는 **setpoint형 mean-diff(Ms) + 기존 conceptor(A)**만. 제거형(ablation-to-zero)은 제외.
  - **WA-LQR(W)**: 타당성 게이트(24a §5) 통과 시 추가 시도.
  - 축은 **within-instruction + cross-scene**만. **cross-instruction 유예** (steering fit/eval 축 기준. exp4-2 B1의 "타 instruction donor 주입"은 실패 *생성* 메커니즘이라 별개 유지).
  - **Task 4종: CloseFridge, OpenDrawer, PPCC-bread, PPCC-beer** (CloseFridge는 신규 cell — 준비 절차 24a §1).

## 1. 배경 (exp3 사후 진단 — 왜 이 두 갈래인가)

exp3(구 pq3) 900판은 전면 null(6-Holm, 위약 대조 포함). 사후 산술 검사(07-20, 배포 NPZ 직접 검사; memory `conceptor-saturation-degenerate`)로 구조가 확정됐다:

- 배포 C_steer는 실제 활성분포 위 R-가중 이득 0.006~0.007 = **≈영행렬** → M=(1−β)I+βC ≈ (1−β)I **균일 감쇠**. perm/gated/null arm이 하는 일의 89~99%가 동일 → exp3의 "위약 대조"는 위약이 아니라 동일 처치였음.
- α를 0.3까지 낮춰도 이득 ~3%(4~5배 회복이나 절대값 미미) → **α 조정으로 못 고침**.
- fit30(2배 데이터)로도 불변 — well-sampled 영역에서 succ/fail **2차 모멘트 부분공간이 실제로 일치**하는 구조적 문제. **자연실패 재fit은 하지 않는다(기결론).**
- 단 "실패 방향이 없다"가 아니다: conceptor(AND-NOT)는 2차 모멘트만 보므로 **평균차 신호를 원리적으로 못 본다**(per-class 중심화가 정확히 그걸 제거). 고정-t 평균차 신호는 별도로 실재(AUROC 0.6~0.7, memory `seen18-genuine-failure-direction`).

exp4의 공략:

1. **exp4-1**: 실패가 (env_name, scenario_seed, inference_seed, 머신)으로 **결정적 재현**됨(patchceil 77/77 bitwise 검증, docs/steering/22)을 이용. 사용자가 실패 영상을 보고 지정한 개입 시점 t0부터 steering을 켜 구제율 측정 — 온라인 검출기 없는 oracle 타이밍, "검출만 되면 구제가 되는가"의 상한 탐색. 연산자 3종 비교: 기존 conceptor / **mean-diff(평균차) 연산자** / exp4-2 산출 conceptor.
2. **exp4-2**: 자연 실패 대신 (Track P) 물리 섭동, (Track I) activation 주입으로 실패를 **만들어** succ/fail 분포를 벌린다. 가설: 분포가 진짜 벌어지면 AND-NOT이 비퇴화 conceptor를 산출한다. 부가 가치: 주입이 pathway별이라 유도 실패에 **TYPE ground-truth 라벨**(goal vs motor)이 설계상 부여됨 — online type 식별의 없던 정답지. 직접 선행: WA-LQR(arXiv:2607.14943, clean-vs-perturbed 대조로 DiT residual diff-of-means steering, 교란 하 SR +11~40pp — 단 위약·유의성 검정 없음; 노트 docs/references/reading_notes/steering_robustness_wam_lqr.md).

## 2. 연산자·arm 용어 (두 세션 공통)

| 기호 | 연산자 | 식 | 출처·비고 |
|---|---|---|---|
| **A** | 기존 exp3 conceptor (legacy) | h′ = h·Mᵀ, M = (1−β)I + βC_steer (**알려진 상태: 데이터 위 ≈(1−β)I**) | exp3 배포 NPZ 그대로 (재fit 없음). "감쇠 단독 대조" 해석 금지(정확한 (1−β)I 아님·dose 상이 — Gate1 반영) |
| **Ms** | setpoint형 mean-diff (**primary**) | h′ = h − β[(h·r̂) − s]·r̂, r̂ = normalize(μ_fail − μ_succ) 비중심화 fit, s = μ_succ·r̂ | affine(= (I−βr̂r̂ᵀ)h + βs·r̂). 선행: ACE(2411.09003)·LEACE·WA-LQR setpoint. **serve affine hook 확장 ~120 LOC 필요** (24a §4.1) |
| **Pr** | 위약: label-permutation setpoint | Ms와 동형, r̂_perm = 라벨 순열 fit | 무작위 방향은 dose-matched 아님(Gate1 반영) — held-out ‖Δh‖ 분포 일치 확인 후 동결 |
| **W** | WA-LQR closed-loop (조건부) | 층별 u = V_out·K·(α·v), α = λμ − v·z | 타당성 게이트 통과 시만 (24a §5, 참고 24c) |
| **B** | exp4-2 유도실패 conceptor | h′ = h·Mᵀ, M = (1−β)I + βC_steer(induced) | exp4-2 산출물 (§4 계약) |
| **A0** | no-steer | — (hook 미등록) | 결정론 재확인 (sentinel 방식, 24a §8) |

- 제거형(I − βr̂r̂ᵀ 단독)은 사용자 결정으로 제외 — Ms에서 s≈0이면 동치이므로 fit 시 s 값 보고로 갈음.

## 3. 비퇴화 진단 (신규 `steer/diag_conceptor_nondegen.py`, CPU)

07-20 산술 검사의 스크립트화. 입력 NPZ + held activation, 출력 JSON:
1. C_steer(또는 I−r̂r̂ᵀ)의 고유값 스펙트럼·effective rank,
2. **R-가중 이득** (C_success에서 R 복원, exp3 기준값 0.006~0.007과 비교),
3. on-data 효과비 r(h)=‖(M−I)h‖/‖h‖ 분포 + 변화 방향의 구조성(PCA 집중도).

적용 규칙:
- **exp4-2 산출 conceptor(B)와 모든 신규 fit: SR eval 투입 전 하드 게이트.** 유도실패 fit의 R-가중 이득이 자연실패 fit(~0.006) 대비 유의하게 크지 않으면 그 변형은 SR eval 없이 종료 — 이 진단 자체가 exp4-2의 1차 가설 검정이다.
- **"유의하게 큼"의 정의 (Gate1 반영)**: 이 지표는 fit 산출물에서 R을 복원하는 자기참조적 양이므로 단독 검정이 아니다 — **held-out episode + label-permutation null 분포**로 임계값을 산출해 sanity gate로만 쓴다 (인과 주장 아님).
- α 그리드는 0.3 아래로 확장, C_succ 이득이 1에 붙으면 포화로 기각.
- arm A는 기진단(퇴화 확인)이라 면제 — 사용은 사용자 결정(legacy 기준선 역할).
- Ms/Pr은 구성상 rank-1이라 퇴화 불가능하지만 r̂ 추정 안정성(부트스트랩 각도 분산)과 s 값을 같은 JSON에 기록.

## 4. exp4-2 → exp4-1 인터페이스 계약

- exp4-2 fit 산출물 배치: `<base_B>/steer/dit_L{L}/{conceptors.npz, metadata.json}` (키 `alpha{a}_C_steer` — fit_phase_conceptor_n15.py 기존 출력 규약) + 비퇴화 진단 JSON 동봉.
- exp4-1 serve가 `--steering-phase-npz-base <base_B> --steering-phases steer`로 로드. exp4-1은 B 없이 먼저 시작하고 NPZ 도착 시 slot-in.
- Ms/Pr NPZ는 벡터 규약(`alpha0_v_steer` [D] + `alpha0_s` 스칼라, affine hook 전용 키 — 24a §4.1)으로 같은 디렉토리 구조에 배치.

## 5. 자원·세션 규칙

- **GPU 규칙(2026-07-21 개정)**: 고정 할당(구 "4 5 6") 폐기. **비어 있는 GPU를 쓴다** — 세션 시작 시 nvidia-smi 확인 후 선점 선언(MACHINE.txt/큐 방식). 고정 규칙은 **GPU 1개당 serve 2개**(로컬 16GB; A100 worker2는 serve 6)뿐.
- 두 세션이 같은 GPU·포트를 잡지 않도록 시작 시 상호 확인. 포트 공유 금지(pkill 포트 패턴 사고 이력).
- 공통 표준: EVAL_SEED=100000, fresh process per episode(gym.make 연속 생성 금지), setsid detach, CPU ≤40%(OMP/OPENBLAS ≤16), 수집 캡처는 exp4-2만 ON(eval은 OFF 정책 유지).
- 결정론은 **머신 단위** — episode 재현·eval은 그 episode를 수집한 머신에서. cross-machine 비교는 각주 규칙(memory `a100-worker2-parallel-eval`).
- 브랜치: exp4-1 `exp/exp4-1-oracle-rescue`, exp4-2 `exp/exp4-2-induced-failures` (dev 분기). patchceil worktree(`.claude/worktrees/patching-ceiling`)는 **포팅 원본으로만, 무접촉**.
- **동시 실행 규칙 (같은 머신에서 두 세션 병행 시)**: git 폴더 하나에는 브랜치 하나만 체크아웃되므로 두 세션이 본 트리를 공유할 수 없다 → **exp4-1은 본 트리, exp4-2는 전용 worktree**(`.claude/worktrees/` 밑, 자기 브랜치)에서 작업한다. worktree는 repo 안이라 robocasa 컨테이너 mount에 그대로 보이며, worktree 내 serve·수집은 patchceil에서 기검증(exp3와 병행 실적).
- 보고는 confound-audit skill 경유, 판정은 pre-registered primary contrast만. 결과 문서 번호는 작성 시점의 dev 문서 목록을 보고 지정한다 (25번대는 patchceil 결과·CloseFridge 라벨러가 이미 사용 — 이 계획 작성 후 dev에서 번호 재편 있었음).

## 6. 게이트 흐름 (요약)

```
[공통] Codex Gate1 완료(07-22, 1라운드) → 문서 확정 커밋
[exp4-1] cell 준비(주석팩·사이드카·CloseFridge C0) → 사용자 t0 지정(동결) → smoke 5종
          → A0 sentinel → 본 eval (A/Ms/Pr) → WA-LQR F1 통과 시 W 추가 → B 도착 시 slot-in
[exp4-2] 코드 2트랙 → smoke S1-S5 → P0 파일럿 → 실패율 게이트(40-70%) → 다리(bridge) 게이트
          → P1 본수집 → fit + 비퇴화 진단(=1차 검정) → 통과분만 분리/transfer/TYPE 평가 + exp4-1 전달
```
