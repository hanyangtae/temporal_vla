# KAI2026 논문 핸드오프 (2026-08-20)

새 세션이 이 문서만 읽고 논문 작업을 이어갈 수 있게 정리한 단일 출처.
브랜치 **`feat/kai-paper`** (origin push 완료), 산출물 전부 `manuscript/`.

## 0. 투고 정보

- **제7회 한국 인공지능 학술대회** (KICS 인공지능소사이어티, koreaai.org)
- 제출 마감 **2026-09-04**, 심사 후 최종본 9/12, 최대 **2쪽** (현재 3쪽 — 마감 전 축약 예정, 사용자 승인됨)
- 한글 작성 확정 (양식이 한글 샘플 배포·"영어논문"은 별도 트랙). 제목·저자·소속 영문 병기는 양식 요구라 이미 있음
- 제출: ManuscriptLink, PDF (사용자가 Overleaf에서 XeLaTeX 컴파일 — kotex 필요)
- 저자: 박경태(1), 김상우(2), 오윤선(교신 †), 한양대학교. TODO: 교신저자 이메일, 학과 표기, 영문 철자 확인, 사사

## 1. 논문 주장 (확정된 프레이밍)

- **메인**: VLA 내부 activation만으로 현재 action phase를 읽을 수 있다.
  판독 정확도(학습에 쓰지 않은 scene): **비지도 군집 0.86 / 지도 probe 0.89** vs
  외부 대조군(시간 0.61–0.68, 정책 행동 0.56–0.61, 행동+시간 결합 0.77)
- 서론 프레임 = **XAI**: "모델이 스스로 무엇을 하고 있다고 표상하는가"는 activation을
  직접 읽어야만 알 수 있다 (외부 관측은 겉보기 행동까지만) — 사용자 지시로 확정
- 보조: ① cluster는 phase-순수(purity 0.888, contingency 블록대각) ② scene 잔차화 후에도
  margin 유지(scene 암기 아님) ③ AE↔SAE·신규 데이터 재현(z +8.8~13.5 / +4.3~5.1)
  ④ k 강건성(k≥8 평평)
- **강등된 주장**: "사람 phase보다 세밀하다"는 메인에서 내려 관찰 문단으로만
  (k 산술 반박 근거는 유지: k=8에서도 10/10 구간 짧음·전환수>k−1 재방문)
- **금지 표현** (사용자·실측 합의): "드문 phase일수록 우위"(편차 큼 — macro-F1 우위까지만),
  ΔSR 개선 주장, "1:1 대응/겹치지 않음", "교차검증", 41 수치의 단독 confirmatory 인용

## 2. 확정 파이프라인·프로토콜

- 데이터: RoboCasa, GR00T N1.5, **5 task / 10 instruction / 930 에피소드**(scene 10 × noise 10,
  apple만 30) = 89,766 스텝. **스텝 = 정책 추론 1회 = 환경 5스텝** (원고에 정의 문장 있음)
- 특징: DiT residual layer 12 · denoise 3 · 49토큰 평균 = 1536d
- 파이프라인(메인): **raw-1536 → AE(latent 16) → KMeans** (PCA 없음).
  AE 규격 = 동료 코드 실측(hidden 256×2 GELU, 대각 가우시안 NLL, AdamW 1e-3/wd 1e-4,
  grad clip 5.0, early stop). 근거: 세 파이프라인 비교에서 raw+AE 최고
  (margin 0.343·10/10 양수·purity 0.875 > raw 직결 > PCA-64w)
- 판독 프로토콜: **scene 단위 train/test 분리**(test 2 scene, split seed 0/1/2 평균),
  군집·최빈 phase 매핑·probe 전부 train scene에서만 적합. 라벨은 군집→phase 매핑에만 사용
- 용어: "held-out" 대신 **"학습에 쓰지 않은 scene(unseen scene)"** (사용자 지시)

## 3. 핵심 수치 (정본 = `manuscript/numbers.md`, 수치 수정은 반드시 그 표 경유)

판독 정확도 (instruction 중앙값):

| 방법 | acc | macro-F1 |
|---|---|---|
| 다수 클래스 | 0.561 | 0.231 |
| 시간(절대 스텝 t) | 0.675 | 0.307 |
| 정책 행동 probe | 0.614 | 0.339 |
| 행동+시간 probe(최강 외부) | 0.773 | 0.559 |
| **activation 비지도 군집** | **0.856** | **0.570** |
| **activation probe** | **0.893** | **0.721** |

- 같은 지도 조건(probe vs 행동+시간): activation 승 **9/10** (예외 OpenDrawer left −0.09)
- k sweep(비지도 acc): k4 0.62 / k6 0.80 / **k8 0.86** / k12–32 0.88–0.89 → 특정 k에 안 기댐
- 잔차화: mi_scene 0.44→0.27, margin 10/10 양수 유지(+0.34→+0.31)

## 4. 파일 지도 (`manuscript/`)

| 파일 | 내용 |
|---|---|
| `main.tex` | 원고 본체. **개정분은 `\new{}`(빨강) 표시** — 제출 전 preamble의 `\newcommand{\new}[1]{\textcolor{red}{#1}}`를 `{#1}`로 바꾸면 사라짐 |
| `numbers.md` | ★수치↔정본 대조표 (금지 표현 경고 포함) |
| `references.md` | 서지 9편 전부 실존 검증(확인 경로 명시). N1.5는 N1(2503.14734) 인용+"후속 릴리스" 서술 |
| `baseline_survey.md` / `eval_practice_survey.md` | 비교 논문·eval 관행 조사 (LAR-MoE가 최근접 경쟁, 직접 경쟁 0편) |
| `figs/` + `make_fig*.py` | Fig1 contingency+잔차화(`make_fig1.py`), Fig2 k sweep(`make_fig_ksweep.py`), Fig3 판독 정확도(`make_fig_readout.py`). 구 fig2(구간길이·방법비교)는 폐기됨 |
| `ref/` | 실험 산출 JSON/TSV 추적본 (align/resid/contingency/k_sweep/readout) |
| `preview_local.pdf` | 로컬 tectonic 렌더 (한글 줄바꿈만 Overleaf와 미세 차이) |
| `README.md` | Overleaf 절차 |

- **본문 인용은 [n] 리터럴** (참고문헌 번호 나중에 정리하기로 사용자 결정 — 현재 서론에 번호 없음)
- 로컬 컴파일 검증법: `sed 's/\\usepackage{kotex}/&\\XeTeXlinebreaklocale ""/' main.tex > _v.tex`
  → tectonic(0.15.0 musl 바이너리) 컴파일 → pymupdf로 PNG 렌더 후 눈검. tectonic·pymupdf는
  세션 tmp에 있었으므로 재다운로드 필요할 수 있음

## 5. 실험 코드 (전부 `scripts/analysis/grid_phase/`, feat/kai-paper 커밋됨)

| 스크립트 | 역할 |
|---|---|
| `intrinsic_phase.py` | 파이프라인·지표 정본 (mi_bits/purity/clock_clusters/margin/boundary_f1, exp/grid-phase-sep에서 이식) |
| `ae_cluster.py` | raw→AE→KMeans 본체. `--dump-labels`로 per-record 라벨+latent+centers NPZ 덤프 |
| `paper_supplements.py` | 게이트(기준 재현 bit-identical 확인됨)·scene 잔차화·contingency. `--pca-dim/--no-whiten` 지원 |
| `phase_readout.py` | ★판독 실험 본체 (5개 대조군 포함, numpy만) |
| `twfinch_baseline.py` | TW-FINCH(CVPR 2021) 저자 코드 무수정 호출 대조군 — 저자 twfinch.py의 191행 미정의 변수 버그로 일부 에피소드 실패(중앙값 latent 0.602/action 0.537, 보조용) |
| `render_phase_timeline.py` / `render_activation_space.py` | 시연 영상 렌더러 (장면 무손상 assert 내장) |

원격 실행 패턴: 스크립트를 ssh stdin으로 승준 `/tmp`에 놓고 `~/anaconda3/bin/python`(torch 있음,
sklearn/scipy 없음)으로 실행, 소용량 결과만 회수. shard 데이터 =
`~/datasets/temporal_vla_store/groot/n15/analysis/grid_phase/segA/*.npz` (30GB, 실존 확인).
KMeans 게이트 재현에는 `--kmeans-src ~/workspace/temporal_vla/task_classification/phase/clustering/gpu.py` 필수.

## 6. 진행 중이던 작업 — Event-SAE 대조군 완전 재현 (사용자 지시: "완전 재현해봐")

근거: 사용자가 "대조군은 기존 논문 방식을 그대로" 요구 → Event-Grounded SAE
(Jin et al., Purdue, 로컬 PDF `docs/references/Event-Grounded Sparse….pdf`,
코드 github.com/xc-j/Event-SAE)를 완전 재현하기로 결정.

**확정 사양** (저자 코드·부록 Table 5b 실측):
- keyframe: AWE `dp_waypoint_selection` pos_only, **η=0.05**
  (AWE 원본 github.com/lucys0/awe → `waypoint_extraction/extract_waypoints.py:158`)
- descriptor: `[1.0·L2(vision) ‖ 0.5·L2(zscore(state)) ‖ 0.4·zscore(progress)]` → 행 L2
  (Event-SAE `event_sae/events/cluster.py:46 build_task_vectors`)
- vision = **`google/siglip-base-patch16-224`** `get_image_features`, keyframe 주변
  5프레임 strip 평균 (`events/build_features.py`) — 로컬 transformers 5.3.0으로 가능
- state = eef_pos(3) (+ gripper_action) (`state_vector_from_record`)
- 군집: sklearn `AgglomerativeClustering(metric="cosine", distance_threshold=0.18)`,
  task-local, **커버리지 0.5 이상 군집만 채택**
- 평가 연결: 얻은 event 군집을 우리 판독 틀에 넣어 phase 정확도 비교 (저자는 SAE feature
  랭킹용 — 용도 차이를 논문에 명시할 것)

**재료 준비 상태**:
- 대상은 우리 실험과 **완전히 같은 930 에피소드** (사용자 지시). 목록 = 세션 tmp
  `episode_keys.json`(슬러그·scene·noise 930개 — 없으면 labels NPZ에서 재생성 가능)
- eef 상태 추출: 승준 `/tmp/extract_states.py` → `/tmp/kai_states/*.npz`
  (eef_pos/eef_quat/gripper_qpos/success, 930판 대상, 실행 중이었음 — `STATES_DONE` 확인 후 회수)
- 영상 930개: 로컬 세션 tmp `vids930/`에 다운로드 완료(휘발) — 필요시 재수신
- 저자 코드 clone: 세션 tmp `Event-SAE/`, `AWE/` (휘발 — 재clone 간단)
- 참고: 판독 라벨·latent NPZ는 승준 `/tmp/kai_lab/out/labels_*_k8.npz`에도 있음(휘발성 /tmp 주의)

**주의**: OvenRack_out 100판·PPCC_marshmallow 11판의 영상은 시각 임베딩에 쓸 수 없는
품질 문제가 있어(별도 트랙에서 재생성 중) — Event-SAE 대조군의 vision 성분은 우선
깨끗한 8개 instruction으로 돌리고, 해당 2개는 재생성본이 나오면 채울 것.

## 7. 남은 일 (우선순위)

1. Event-SAE 대조군 완주 → 판독 표·Fig3에 추가, 방법 절에 인용 문장
   ("PACE류 출력 기반·Event-SAE 이벤트 군집을 같은 평가 틀에서 대조군으로")
2. TW-FINCH 결과를 보조로 병기할지 결정(저자 코드 버그 처리 방식 명시 필요)
3. 참고문헌 번호 정리 (서론이 현재 무번호 — references.md 순서로 부여)
4. 2쪽 축약 (우선순위: 참고문헌 압축 → 그림 취사선택 → 결과 문단 압축)
5. `\new` 빨강 제거는 사용자가 diff 확인한 뒤
6. 저자 정보 TODO 채우기 (이메일·학과·사사)

## 8. 세션 운영 규칙 (이 작업에 한정된 것)

- 사용자 직접 작성 문단(요약·서론 대부분)은 **덮어쓰지 말 것** — 컴파일 깨지는 특수문자만
  이스케이프 허용. 사용자가 파일을 동시에 편집하니 편집 전 git 상태 확인
- 개정 표시: 새로 쓰는 문장은 `\new{}`로 감싸기 (사용자가 diff를 색으로 봄)
- 수치는 numbers.md 경유(verify-before-relay), 서지는 실존 확인된 것만
- main repo 작업 트리를 다른 세션들과 공유 중 — 브랜치 전환 금지, 필요시 임시 worktree
