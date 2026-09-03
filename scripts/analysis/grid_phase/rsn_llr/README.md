# rsN_llr 라운드 분석·fit 스크립트 (연산자 설계 세션, 2026-08-31 ~ 09-02)

전부 승준 원격(`~/anaconda3/bin/python`, OMP≤12)에서 ssh-stdin/scp로 실행하던 probe·fit.
결과 정본 = Notion `3c363918d42a8029910ec141eb83f988` (수식·표·판정) +
`docs/collab_within_claude/handoff_20260902_v4r_round.md` 계열. 데이터 경로는 승준
`~/datasets/temporal_vla_store/groot/n15/analysis/` 기준 (스크립트 안 상대 규약 참조).

## 데이터 검토 배터리 (grid v2, segB_v2 캐시)

| 파일 | 내용 |
|---|---|
| `segb_extract.py` | grid v2 전 격자 pkl → task별 압축 캐시(L12·last denoise·토큰 mean + proprio18) |
| `segb_battery.py` / `segb_battery_mixed.py` | read 배터리(condg margin·부분공간·mean-diff·길이) — 전 scene / 혼합 scene 한정 |
| `segb_space.py` / `segb_space2.py` | 특징공간 비교 raw-1536 vs PCA-16 (vs AE-16) — 가우시안 대조 채점 형태 확정 근거 |
| `segb_situfit.py` / `segb_situ16.py` | scene-전용 fit(15판) vs pooled 비교 — "15판 부족" 실측 |
| `segb_clean.py` | scene 내 분리 존재 증명 (완전 held-out) — 컨택-후 0.83 |
| `segb_clucmp.py` | 채점 신호 4-arm 정면 비교 (LLR vs proprio-잔차 vs cluster소속 vs hybrid) |
| `segb_gridmap.py` / `segb_heatmap.py` | scene×noise 성패 지도·heatmap 시각화 |

## rsN_llr 등록·번들 (v4/v4r, segA 계열 캐시)

| 파일 | 내용 |
|---|---|
| `v4_reg2.py` | v4 등록표 + LLR 번들(NPZ serve 계약: scaler/enc/succ_mean/mu·cov/ood_lo) |
| `v4_occ.py` | 등록 cluster의 실패-record 점유율 — "read의 거처 ≠ 발화의 거처" 발견 |
| `ood_discrim.py` | OOD 전원기각 판독 분리(fit-창 통과 여부·encode 일치 검증) |
| `recal_oodlo.py` | ood_lo 발화-분포 재보정 (실패 후반절반 5퍼센타일, scorer 스케일) |
| `v4_fit_pool3.py` | v4sb pool 규약 fit(setM+LLR) — 게이트=scene-내 CV 절대 기준(≥0.70) |
| `v4_fit_pool4.py` | v4r ck8판 (segA_v4r_ck8 소비, instr_setm_v4r_ck8 + rsn_llr_reg_v4r 산출) |
| `v4r_setm_gt.py` | setm_gt v4r판 — 대상 scene 실패만 v4r pkl(gt_phases)로 교체 |

주의: `fix_oodlo.py`(상수항 −8·log2π 보정)는 일회성 패치라 미수록 — 그 보정은
`v4_fit_pool3.py` 이후 세대의 `gll`(2π 상수 포함)에 흡수됨. ood_lo 스케일은 반드시
serve `llr_scorer.py`의 logpdf(상수항 포함)와 같은 정의로 산출할 것.

## v5 라운드 (scene-local LOKO, 2026-09-03~)

| 파일 | 내용 |
|---|---|
| `v5_fit_loko.py` | (instr, scene, 대상 k)당 연산자 fit — setm gt/ck8 + rsn_llr 번들. pool = 타 k 전판 + 대상 k 실패만(success-blind). 게이트 = k-계층화 concordance + 순열검정(§ 아래) |
| `v5_loko_feasibility.py` | activation 없이 index_v5 만으로 (instr,scene,k)별 pool 성공 ep 수 → 성립 가능성 사전 판정 |

**게이트 설계 근거(반드시 읽을 것)**: LOKO pool 은 성공이 타 k 에서만 오므로 라벨이
지터 k 와 상관된다(k 단독 AUROC 중앙값 0.835). pool 전체 AUROC 로 등록하면 outcome 이
아니라 k 를 읽는 판별기가 통과한다. 등록 기준은 succ·fail 공존 k 안의 쌍만 센
concordance + k 안 라벨 순열검정 p≤0.05 이고, 앵커(k-중심화)는 **LOO 안에서** 잡아야
한다(전체로 잡으면 순수 노이즈에서 0.73~0.97 오검출). 상세·합성 대조 수치 =
`docs/collab_within_claude/handoff_20260903_연산자설계.md` §11.
