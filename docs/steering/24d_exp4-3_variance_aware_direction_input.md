# exp4-3 입력 — 분산+평균 모두 고려한 방향(whitened mean-diff) probe·steer 설계 근거

작성: exp4-1 dashboard 세션(관찰 전용), 2026-07-24.
이 문서는 **exp4-3(분산+평균 모두 고려한 probe/steer) 설계 시 반드시 반영할 dashboard 실측**이다.
계획 확정본이 아니라, 계획 세울 때 넣어야 할 근거·처방·함정 모음.

핵심 한 줄: **succ/fail 판별 이득은 "차원 수"가 아니라 "방향의 질(whitening)"에서 온다.
setM 을 rank-1 유지한 채 방향만 raw mean-diff → whitened mean-diff(Σ⁻¹δ)로 바꾸는 것이 첫 처방.**

---

## 0. 어디서 나온 이야기인가 (맥락)

exp4-1 에서 rank-1 affine 개입(**setM**: h' = h − β[(h·r̂)−s]r̂, r̂ = normalize(μ_fail−μ_succ))이
어떤 처치도 위약·노이즈 재추첨을 못 넘겼다. 그 원인을 진단하려고 dashboard 가 **관찰(read-out)
프로브**를 돌렸다 — "표현 안에 succ/fail 정보가 얼마나, 어떤 형태로 있나"를 개입 없이 측정.

프로브는 exp4-1 setM 이 실제로 fit 한 그 표본·layer·길이통제 cap 을 그대로 쓰고(게이트: 재현한
mean-diff CV AUROC 가 fit 기록과 일치해야 통과 — drawer_right L4 에서 0.752 = fit 기록 0.752
정확 일치), episode 단위 5-fold + 순열 null 로 검정했다. 스크립트/데이터는 §5.

---

## 1. 핵심 실측 (exp4-3 가 딛고 설 근거)

### 1.1 비선형은 이득이 없다
5개 cell 전부에서 커널SVM·MLP·k-NN 이 다차원 **선형**(logreg)을 유의하게 못 넘었다.
→ succ/fail 정보는 비선형으로 꼬여 있지 않다. **커널/오토인코더/SAE 로 갈 이유가 이 데이터엔 없다.**

### 1.2 다차원도 필요 없다 — rank-1 로 충분
"다차원 학습선형(logreg, PCA50 공간)"이 "rank-1 whitened 방향(LDA = Σ⁻¹δ)"을 **못 넘는다**.
전 cell 에서 `logreg − lda ≤ 0`:

| cell | setM(raw δ) | **LDA(whitened δ, rank-1)** | logreg(다차원) | logreg−lda |
|---|---|---|---|---|
| drawer_L | 0.674 | **0.885** | 0.870 | −0.015 |
| drawer_R | 0.752 | **0.921** | 0.915 | −0.006 |
| bread | 0.735 | **0.812** | 0.804 | −0.008 |
| beer⚠ | 0.666 | 0.894 | 0.853 | −0.041 |
| mixer✗ | 0.615 | 0.630 | 0.601 | −0.028 |

(⚠ beer = 게이트 어긋남: 구 manifest `task_PPCC_fit` 사용, raw δ 0.666 이 fit 기록 0.768 과 불일치.
`task_PPCC_fit_beerclean` 로 재실행 필요. ✗ mixer = 신호 자체 약함, LDA z=+1.28 p=0.484 비유의.)

→ 최종 개입축은 어느 프로브든 **원공간 벡터 1개**(rank-1)로 환원된다. 차이는 방향의 질뿐.

### 1.3 이득의 정체 = whitening (분산 정규화)
raw δ 를 (a) 차원축소만 / (b) 차원축소+whitening 으로 분해:

| cell | raw δ (setM) | +차원축소 (PCA raw δ) | +whitening (LDA) | 차원축소분 | **whitening분** |
|---|---|---|---|---|---|
| drawer_L | 0.674 | 0.704 | 0.885 | +0.030 | **+0.181** |
| drawer_R | 0.752 | 0.780 | 0.921 | +0.028 | **+0.141** |
| bread | 0.735 | 0.746 | 0.812 | +0.011 | **+0.066** |

이득의 **거의 전부가 whitening**에서 온다. 차원축소만으론 +0.01~0.03. 순열검정: LDA 의
setM 대비 이득이 drawer_L/R 에서 p=0.032(유의), bread +0.077 은 있으나 p=0.226(약).

### 1.4 이것이 말하는 setM 실패의 유력한 기술적 원인
raw δ 와 whitened δ 가 이만큼 갈린다 = **μ_fail−μ_succ 방향이 "분산이 큰 축"(scene·phase
변동)을 향하고 있었다.** setM 은 그 고분산 축을 setpoint 로 뭉개고, 정작 일관되게 갈리는 진짜
판별축은 안 건드렸을 개연성. → "쓰기가 원래 안 된다"가 아니라 **"엉뚱한 축을 밀고 있었다"** 는,
고칠 수 있는 결함. (단 이건 read 근거. write 확인은 exp4-3 SR eval — §4.)

---

## 2. exp4-3 설계 처방

### 2.1 probe (판별기)
- **1순위 = whitened mean-diff (LDA):** w = Σ_within⁻¹ (μ_fail − μ_succ), **shrinkage 필수**.
  Σ_within = pooled within-class covariance. shrinkage: Σ_s = (1−λ)Σ + λ·(trΣ/D)·I.
  현재 프로브는 PCA-50 공간 + λ=0.1(임의). **exp4-3 는 λ 와 PCA 차원(또는 Ledoit-Wolf 자동
  shrinkage)의 민감도를 사전등록 검정으로 고정할 것.**
- 비선형/다차원 프로브는 **상한 대조로만** 두면 충분(1.1·1.2 에서 이득 없음 확인). 새로 크게
  안 짜도 됨.

### 2.2 steer (개입 연산자)
- **setM 의 방향만 교체**: r̂ = normalize(raw δ) → r̂ = normalize(Σ_within⁻¹ δ).
  나머지 형태(h' = h − β[(h·r̂)−s]r̂, s = μ_succ·r̂, β=1)는 유지 가능.
- **여전히 rank-1** → 분산 파괴가 1축만. exp4-1 에서 걱정한 "다차원 setpoint 붕괴(k차원을
  한 점으로 collapse → 분산 파괴 k배)"를 **하지 말 것.** 1.2 에서 다차원 불필요가 확인됐으니
  naive 다차원 부분공간 setpoint 는 손해만 크다.
- setpoint 좌표/사영을 원공간에서 할지 whitened 공간에서 할지는 설계 결정. 원공간 벡터 1개로
  project 하는 편이 setM 배선(serve SetpointSteering)과 호환. whitened 방향을 원공간 단위벡터로
  정규화해 넣으면 기존 hook 그대로 재사용 가능(§5 배선 참고).

### 2.3 conceptor 와의 관계 — "분산+평균 모두"의 정확한 형태
- exp3/exp4-1 conceptor(C_steer = C_success ∧ ¬C_fail)는 **2차 모멘트(공분산) 대조만** 쓰고
  평균차를 명시적으로 안 본다. 게다가 전 cell·전 layer 퇴화(|z|<2, M≈(1−β)I).
- LDA(Σ⁻¹δ)는 **1차 모멘트(평균차)를 분산으로 정규화** = "평균과 분산을 모두" 쓰는 정확한 최소
  형태. exp4-3 가 겨냥하는 게 바로 이것. conceptor 의 다차원성이 필요했던 게 아니라, **평균차의
  분산-정규화**가 필요했던 것으로 이 데이터는 가리킨다.
- 따라서 exp4-3 는 conceptor 를 버리는 게 아니라 "conceptor 가 하려던 분산 고려를 평균차 축에서
  하는" 최소 버전으로 재정식화하는 셈.

---

## 3. 반드시 함께 통제할 함정

- **scene confound (미해결):** fit30·natural_strict 는 scene(scenario_seed)당 rollout 1개라
  succ scene 과 fail scene 이 애초에 다른 물리 상황이다. 30 scene 전부 "한쪽 결과만". 그래서
  판별 이득(+whitening 포함)이 "실패의 인과 신호"인지 "어려운 scene 의 겉모습"인지 read 로는
  못 가른다(episode AUROC 0.95~1.00 이 그 서명). 같은 scene 에서 succ/fail 갈리는 유일 데이터
  (노이즈 재추첨)는 **활성 캡처 OFF** 라 대조 불가. → exp4-3 probe 결과를 "인과"로 부르지 말 것.
  진짜 통제는 재추첨 판 활성 재수집(별건, 4-1-1 세션에서 t0 미리읽기 검정으로 진행 중).
- **read ≠ write:** §1·§2 는 전부 read-out 근거. whitened 방향이 개입에 먹히는지는 exp4-3 의
  SR eval 로만 답한다. (단 이 각주는 결과 해석 때 한 번 달면 되고, 개선 시도 자체를 막지 말 것 —
  "쓰기가 원래 안 된다"는 합리적 개선을 다 소진한 뒤의 종착 결론이지 시작 전 전제가 아니다.)
- **길이통제:** cap = ceil(μ+1σ) of 성공 episode record 수. fit_mean_diff 규약과 동일하게 유지
  (실패=timeout 과대가중 차단). phase 조건부로 갈 땐 phase 별 dwell cap.
- **평가 표준:** SR eval 은 EVAL_SEED=100000, 처치-위약 쌍으로 실행, 분모 held-out 분리
  (eval-풀 primary / fit-풀 LOO secondary). exp4-1 집계 규약 그대로.
- **beer 게이트:** whitening 이득 수치는 beer 를 beerclean manifest 로 재실행해 raw δ 를
  0.768 근처로 올린 뒤 확정. rank-1 결론 자체는 안 흔들리나 이득 크기엔 필요.

---

## 4. 검증 사다리 (exp4-1 사다리식 ablation 계승)

1. **read (완료, 이 문서):** whitened rank-1 이 raw rank-1 대비 판별 이득 → 확인(drawer_L/R p=0.032).
2. **read 확장:** λ·PCA 차원 민감도 + beer 정정 + 전 cell(mixer 포함) 재현.
3. **write 파일럿:** whitened-setM 을 소수 cell(drawer_L/R 우선 — 신호 강)에 개입, SR ΔSR 대
   raw-setM·위약·노이즈 재추첨. 여기서 raw 대비 개선 없으면 "방향 교체로도 안 됨" = write 결함이
   방향이 아님을 시사(그때 §3 read≠write 각주 발동).
4. **write 확대/조건부:** 파일럿에서 신호 시 phase 조건부·전 cell.

각 단계 이전이 신호를 보일 때만 다음으로(약한 신호에서 복잡도 추가 금지 — exp4 공통 원칙).

---

## 5. 재현 정보 (프로브 자산)

- 스크립트: `scripts/... (dashboard tmp)` → **정착 위치는 exp4-3 담당이
  `scripts/safe/groot_n15/robocasa/steer/exp4_3/probe_whitened.py` 로 이관 권장.** 현재 실체는
  dashboard job tmp 의 `probe_nonlinear.py`(승준 `~/dash_probe/probe_nonlinear.py`).
- 프로브 방향: `meandiff`(원공간 raw δ=setM), `pca_meandiff`(PCA raw δ), `lda`(PCA whitened δ),
  `logreg`/`svm_rbf`/`mlp`/`knn`(상한 대조).
- 선정 layer(setM activation 분리도 기준, layer_sweep.json): beer L10 · bread L10 · drawer_L L2 ·
  drawer_R L4 · mixer L15. cap: 77/50/51/76/62.
- 데이터: fit30 full-token pkl(승준 `~/datasets/.../manifests_fit30/`), token·denoise 평균 →
  [n,D] 후 layer 인덱싱. 캐시 npz(축약본) 승준 `~/dash_probe/cache/<cell>_L<layer>.npz`.
- 실행: 승준 `~/anaconda3/bin/python`(sklearn 1.2.2·scipy 1.11.4·torch 2.1.0 **보유** — 기존
  memory "scipy 없음"은 정정 대상). CPU only, cell 당 캐시 히트 시 순열 30회 ~3분.
- 게이트: `--expect-cap`, `--expect-auroc` 로 mean-diff CV AUROC 가 fit 기록 재현하는지 확인 후
  본 분석. (재현 실패 시 표본/layer/cap 어긋남 의심 — 채택 금지.)

---

## 6. 한 문단 요약 (exp4-3 계획서 상단에 붙일 것)

exp4-1 의 rank-1 setM(raw mean-diff)이 실패했고, dashboard read 프로브는 그 원인이 "차원 부족"이
아니라 "방향이 고분산 축을 향함"임을 가리킨다. 같은 rank-1 이라도 방향을 whitened mean-diff
(Σ⁻¹δ, LDA)로 바꾸면 판별 AUROC 가 drawer 계열에서 +0.14~0.18(p=0.032) 오르고, 다차원·비선형
프로브는 이 rank-1 whitened 를 못 넘는다. 따라서 exp4-3 는 **다차원/비선형이 아니라 "평균차의
분산-정규화(whitening)"를 rank-1 로 적용**하는 것을 첫 처방으로 하고, scene confound·read≠write·
shrinkage 민감도를 통제한 채 write 파일럿(drawer_L/R 우선)까지 사다리로 검증한다.
