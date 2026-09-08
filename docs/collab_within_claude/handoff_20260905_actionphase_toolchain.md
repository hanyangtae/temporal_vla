# Handoff 2026-09-05 — action phase 툴체인 (코드 경로·역할·실행법)

작성: action phase 세션. 대상: **이 라인을 이어받아 바로 손을 대야 하는 세션/사람**.

이 문서는 **"무엇을 어디서 어떻게 돌리나"** 만 담는다. 라운드 판정·수치는
[`handoff_20260903_actionphase.md`](handoff_20260903_actionphase.md) §6b(v6 판정) ·
`docs/steering/RESULTS.md` 를 보라.

---

## 0. 30초 요약 — 지금 상태와 다음 한 걸음

- **정본 데이터** = grid **v6** (`08f1c9df8207`, 1,800판). **v1~v5 는 전부 폐기됐다.**
- **정본 번들** = 승준 `analysis/grid_phase_v6/ae_k8/ae_bundle_k8.npz` (11키, 800ep).
  검증 3종 통과(§5) — **소비 세션이 바로 쓸 수 있는 상태다.**
- 파이프는 **전부 승준(원격 CPU)에서** 돈다. 로컬에서 하는 건 코드 편집·발사·감시뿐.
- 새 데이터가 오면 §3 순서(추출→병합→AE→검증→진단), 기존 데이터로 분석만 더 하려면
  §4 진단 도구를 바로 쓴다.

⚠ **2026-09-05 03:5x 이후 승준 SSH(11112) 불통** — ping 은 되는데 포트만 timeout.
산출물은 그 전에 완료·검증됐으므로 유실 없음. 복구 후 §8-1 점검만 하면 된다.

---

## 1. 데이터 좌표 (승준 `kimseungjun@166.104.146.37 -p 11112`)

| 무엇 | 경로 | 내용 |
|---|---|---|
| 아카이브 | `~/datasets/temporal_vla_store/groot/n15/grid/08f1c9df8207/` | 1,800판 rollout.pkl + meta.json. 좌표 `<machine>/<key>/s<sid>/j<jid>/n<nid>/base/` |
| scene shard | `~/datasets/.../analysis/grid_phase_v6/segA_scene/<slug>__s<i>.npz` | **33개** (11키 × 3 scene, 각 50판) |
| 병합본 | `~/datasets/.../analysis/grid_phase_v6/segA/<slug>.npz` | **11개** (각 150판) — **AE 입력은 이것** |
| 번들·라벨 | `~/datasets/.../analysis/grid_phase_v6/ae_k8/` | `ae_bundle_k8.npz` · `labels_<slug>_k8.npz` ×11 · `ae_pertask_k8.json` · `resid_compare_ae.tsv` |
| 임시 번들 | `~/datasets/.../analysis/grid_phase_v6/ae_k8_partial/` | oven-L 단독 학습본. **배관 검증용, 해석 금지** |
| 진단 산출 | `~/workspace/temporal_vla/outputs/analysis/grid_phase_v6_frozen/` | `contam_v6_all.json` · `early_probe.json` · `prerebase_*.tsv` · `postrebase_ALL.tsv` |
| 로그 | `~/workspace/logs/` | `chain{1,3,4}.log` · `actionphase_scene*.log` · `metapatch_scene3.log` |
| 인덱스 | repo `configs/collect/n15_grid_v6_scene_jitter/index_rollouts_v6.tsv` | 1,800행 (dev 0181cd8) |

**slug 규칙**: `grid_instruction` 의 `/`·공백 → `_`. 11키 =
`CoffeeSetupMug` `DishwasherRack_out-left` `DishwasherRack_out-right` `OpenDrawer_left`
`OpenDrawer_right` `OvenRack_out-left` `OvenRack_out-right` `PPCC_bread` `PPCC_candle`
`PPCC_jug` `PPCC_marshmallow`.
**PPCC/apple 은 사용자 지시로 영구 제외** — 그래서 12키가 아니라 11키다. 번들에 apple
centers 가 없으므로 serve 에서 apple 판정을 요청하면 **fail-loud** 한다(의도된 동작).

**원격 실행 환경**: `~/anaconda3/bin/python` (numpy + torch **CPU**, scipy·sklearn 없음).
repo = `~/workspace/temporal_vla` (브랜치로 동기화, **scp 금지**).
**CPU 예산 8코어** — `WORKERS × OMP_THREADS ≤ 8` (러너가 강제, 초과 시 발사 거부).

---

## 2. 코드 지도 — 파일별 역할

전부 `scripts/analysis/grid_phase/` 아래. 브랜치 `exp/actionphase-v5` (→ `eval_whole_pipe_1` 머지됨).

### 2.1 파이프 (데이터를 만드는 것)

| 파일 | 역할 | 언제 |
|---|---|---|
| **`run_actionphase_remote.sh`** (372줄) | **주 러너.** 인덱스에서 instruction 목록·기대 판수를 읽어 shard 추출 → 셀 단위 감사 → (옵션) AE. **격자를 하드코딩하지 않는다 — 인덱스가 계약이다** | 새 shard 를 뽑을 때 항상 |
| `merge_scene_shards.py` (172줄) | scene shard → instruction shard 병합. **재추출 없이** 이어 붙인다 | 한 키의 scene 이 다 모였을 때 |
| `drive_actionphase_incremental.sh` (103줄) | 완료 instruction 폴링 → 자동 추출·임시 AE | 수집이 대량으로 몰릴 때(현재는 통지 기반 수동 발사라 미사용) |
| `extract_grid_matrix.py` (기존, 수정함) | pkl → shard NPZ 축약. 러너가 호출 | 직접 호출할 일은 거의 없다 |
| `ae_cluster.py` (기존) | AE 학습 + instruction 별 KMeans + 번들 export | 러너 경유 또는 직접 |

### 2.2 rebase·키 변경 대응

| 파일 | 역할 |
|---|---|
| `snapshot_archive_fingerprints.py` (190줄) | meta.json 만 읽어 `(instruction, s, j, n, pkl_sha256)` 표 동결 · `--compare` 로 rebase 전후 대조 · `--emit-index` 로 **아카이브→인덱스** 생성(수집 측 인덱서가 막혔을 때 우회로) |
| `rename_swap_keys.py` (184줄) | 키 **맞교환** — shard·병합본·라벨 파일명 + 번들 `centers.<slug>`·`slugs`·`provenance` 일괄. 검증 내장(실패 시 쓰기 전 중단), 번들 `.prerebase` 백업 |
| `patch_shard_meta_keys.py` (158줄) | NPZ 안 `meta_json.instruction` 패치. zip 멤버 **스트림 복사**(shard 가 2~7GB 라 메모리 상수 유지) |

### 2.3 진단 (판정을 만드는 것)

| 파일 | 무엇을 답하나 | 산출 |
|---|---|---|
| `early_record_probe.py` (273줄) | "초기 활성화가 담은 게 실패인가 초기조건(j)인가" — 창별 **j 판독 정확도** · succ/fail AUROC · **j-only 대조군** · **j 잔차화** · `--per-cell` 셀별 AUROC(방향은 타 j 에서 학습) | `early_probe.json` |
| `cluster_contamination.py` (158줄) | "cluster 가 phase 를 담았나 j 를 담았나" — 창별 MI(c;j) vs MI(c;phase) + cluster 별 max-j 점유율 → **`exclude` 플래그** | `contam_v6_all.json` |
| `scene_activity_check.py` (105줄) | "전패 scene 이 관측 死인가 어려운 배치인가" — time_var · across_var · j_acc 를 같은 키의 다른 scene 과 비교 | stdout 표 |

### 2.4 온라인 판정기 (serve 쪽 — 타 세션 소유, 우리가 계약 검증)

`src/failure_online/cluster_phase.py`:
`ClusterPhaseAssigner.from_bundle(path, task=slug)` → `feature_from_hidden(hidden, layer_idx)`
→ `assign(feat)` / `assign_batch(feats)`. **무상태 per-record**.
`python3 src/failure_online/cluster_phase.py` = 합성 self-test(**배관만** 본다).
**실번들 왕복 검증은 별도로 해야 한다** — §5.1.

---

## 3. 표준 실행 순서 (새 데이터가 왔을 때)

모든 원격 실행은 **`setsid nohup … &`** 로 분리한다. 일반 background 로 돌리면 harness 가
중간에 죽인다(실측: 병합 1건 중단, tmp→replace 라 산출물은 무손상이었다).

### 3.1 추출 — instruction 단위

```bash
ssh -p 11112 kimseungjun@166.104.146.37 \
  "setsid nohup env TAG=v6 INDEX=<인덱스 절대경로> WORKERS=2 OMP_THREADS=4 \
     bash ~/workspace/temporal_vla/scripts/analysis/grid_phase/run_actionphase_remote.sh \
     > ~/workspace/logs/actionphase.log 2>&1 < /dev/null & echo LAUNCHED"
```

### 3.2 추출 — (instruction, scene) 단위 ← **v6 는 이 단위로 완료된다**

위 명령에 `INSTR_SCENES='OpenDrawer/left:0,OvenRack/out-left:1'` (키:scene 콤마 구분)를
추가하면 `segA_scene/<slug>__s<i>.npz` 로 낸다.
**`segA/` 와 분리하는 게 핵심** — 같은 폴더에 두면 `ae_cluster` 가 scene shard 를 별개
instruction 으로 잡아 KMeans 단위가 조용히 바뀐다.

기타 환경변수:
`EXCLUDE_CELLS=<qa_invalid_cells.txt>` (QA 무효셀 제외, 한 행도 안 맞으면 fail-loud) ·
`ALLOW_PARTIAL_PKL=1` (이관 미완인데 강행) · `PARTIAL_AE=1` (모인 shard 만으로 임시 번들,
`ae_k<K>_partial/` 로 분리 산출) · `SKIP_AE=1` · `K=8` · `OUT=<디렉토리>`.

### 3.3 병합 (키의 scene 이 다 모이면)

```bash
~/anaconda3/bin/python scripts/analysis/grid_phase/merge_scene_shards.py \
  --scene-dir <…>/segA_scene --out-dir <…>/segA --require-scenes 3
# --slug <이름> 으로 한 키만 · --dry-run 으로 확인만
```
내부에서 하는 일: scene 오름차순 연결(= `Episode.key()` 순서 재현) · **phase 코드북 union
후 재매핑** · ep_id 재번호 · sigs/merged_from 기록. plan_id·machine 이 scene 간 다르면
행 순서가 깨지므로 fail-loud.

### 3.4 정식 AE (**모든 키가 병합된 뒤 1회**)

```bash
~/anaconda3/bin/python scripts/analysis/grid_phase/ae_cluster.py \
  --shard-dir <…>/segA --mode all --dump-labels --k 8 \
  --epochs 800 --patience 60 \
  --out-dir <…>/ae_k8 --export-bundle <…>/ae_k8/ae_bundle_k8.npz
```
- **`--export-bundle` 없이 돌리면 encoder 가 어디에도 안 남는다**(실사고 이력).
- **200 epoch 는 미수렴** — v6 실측 200ep val 1064 → **800ep 426**. 800 은 11키·162,874
  record 에서 약 45분(승준 CPU 4스레드, epoch 당 ~4.2초).
- AE 는 **shard 파일명(path.stem)으로 slug 를 잡는다** — meta 를 읽지 않는다. 그래서 키
  rename 은 **AE 전에** 해 두면 번들이 처음부터 새 이름으로 나온다(§6-1 참조).

### 3.5 자동 체인 (내가 쓴 것 — `~/workspace/chain*.sh`, git 밖)

`chain1`(추출→병합→활성도) · `chain3`(키 맞교환→병합→AE) · `chain4`(meta 패치→왕복 검증→MI 표).
임시 스크립트라 필요하면 §3 커맨드로 다시 조립하면 된다. 체인 간 연결은
`until grep -q CHAIN3_DONE <log>; do sleep 60; done` 식 폴링으로 걸었다.

---

## 4. 진단 도구 사용법

```bash
# 초기 창이 무엇을 담았나 (+ 셀별)
~/anaconda3/bin/python scripts/analysis/grid_phase/early_record_probe.py \
  --shard <…>/segA_scene/OpenDrawer_left__s0.npz --shard <…>/PPCC_bread__s0.npz \
  --windows 0:1,0:10,30:80 --per-cell --out-json <…>/early_probe.json

# cluster 오염 (labels·shard 를 같은 순서로 짝지어 넘긴다)
~/anaconda3/bin/python scripts/analysis/grid_phase/cluster_contamination.py \
  --labels <ae>/labels_<slug>_k8.npz --shard <segA>/<slug>.npz \
  --windows 0:10,10:30,30:9999 --out-json <…>/contam.json
# 규칙: max-j 점유율 ≥0.40 → exclude=true · record<50 → 보류 · MI(c;j) norm ≥0.10 → 창 경고

# 전패 scene 진단
~/anaconda3/bin/python scripts/analysis/grid_phase/scene_activity_check.py \
  --shard <…>__s0.npz --shard <…>__s1.npz --shard <…>__s2.npz
```

**해석 규칙 (반드시 지킬 것)**
- `purity` 는 **키 간 비교 금지** — phase 종수가 rack 5~6 / PPCC·coffee 3 이라 자동으로 갈린다
  (v6 실측: jug purity 0.970 인데 MI 0.232 · F1 0.036).
- **절대 record 창으로 succ/fail 비교 금지** — 성공 판이 짧아 통째로 빠지는 **생존 판 편향**
  (v6 실측: 전패 scene 판당 144 record vs 전승 37, **4배**). 중·후반은 phase dwell 고정 예산으로.
- `margin` 은 **scene 잔차화 전/후를 나란히** (`resid_compare_ae.tsv`) — v6 에서 rack 계열이
  절반으로 붕괴한 게 이 대조로만 드러났다.
- `MI` 는 과분할에 둔감 → **clock 대조(margin) 없이 raw MI 만 보고 금지**.

---

## 5. 검증 절차 (산출물을 내기 전에 반드시)

### 5.1 온라인 왕복 — **번들을 배포하기 전 필수**

합성 self-test 는 배관만 본다. **실번들 + 실 shard** 로 전 record 일치를 확인한다:

```python
import numpy as np, sys
sys.path.insert(0, "/home/kimseungjun/workspace/temporal_vla")
from src.failure_online.cluster_phase import ClusterPhaseAssigner
L, D, S = 5, 3, 3          # layer 12 · 마지막 denoise · "all"(49토큰 mean)
X   = np.load(SHARD)["X"][:, L, D, S, :].astype(np.float32)
ref = np.load(LABELS)["cluster"]
asg = ClusterPhaseAssigner.from_bundle(BUNDLE, task=SLUG)
got = np.array([o["idx"] for o in asg.assign_batch(X)])
assert (got != ref).sum() == 0
```
v6 실측: 11키 **162,874 record 불일치 0**. (chain4 §2 에 스크립트로 들어 있다.)

### 5.2 파일명 == 내부 meta

```bash
python3 scripts/analysis/grid_phase/patch_shard_meta_keys.py \
  --swap A=B --dir <segA_scene> --dir <segA>      # dry-run 이 곧 검사
```
불일치 목록이 뜬다. `--apply` 하면 패치 후 사후 검증이 자동으로 돈다.

### 5.3 rebase 지문 (아카이브가 rename·이동될 때)

```bash
python3 …/snapshot_archive_fingerprints.py --grid-root <구 루트> --out pre.tsv
# … rebase 실행 …
python3 …/snapshot_archive_fingerprints.py --grid-root <신 루트> --out post.tsv
python3 …/snapshot_archive_fingerprints.py --compare pre.tsv post.tsv \
  --swap OvenRack/out-left=OvenRack/out-right \
  --swap DishwasherRack/out-left=DishwasherRack/out-right
```
**전체 키로 떠라** — 교환 대상뿐 아니라 나머지가 항등으로 남았는지도 같은 표에서 검증된다.
동결은 **rebase 직전에** 해야 한다(그 전에 뜨면 그 사이 수집분이 빠진다).

---

## 6. 함정 (전부 이번 라운드에서 실제로 물린 것)

| # | 함정 | 방어 |
|---|---|---|
| 1 | **rename 시 파일명만 바꾸고 NPZ 내부 meta 를 안 바꿈** → 파일명으로 찾으면 맞고 meta 로 찾으면 **반대쪽 물리 대상**. 에러 없이 틀린 산출물 | `patch_shard_meta_keys.py` 를 **같이** 돌린다. 더 나은 방법: **AE 전에 rename** 하면 번들·labels 가 처음부터 새 이름이라 이 단계 자체가 사라진다 |
| 2 | **맞교환을 순차 치환으로 하면 한쪽이 덮인다**(두 키가 같은 값) | `rename_swap_keys.py` — 전량 사전 복사 후 일괄 배치 + 교환 검증 내장 |
| 3 | **2GB 초과 zip 멤버는 `force_zip64=True` 없이 스트리밍 쓰기 실패** — 합성 소파일 테스트로는 절대 안 잡힌다 | 실제 크기 데이터로 검증 |
| 4 | **감사 기대치를 추출에 쓴 같은 인덱스에서 만들면** 이관 미완(meta 50/pkl 46)이 "46/46 일치"로 통과 | 러너가 추출 **전에** `has_pkl` 대조(exit 14) |
| 5 | **완료 판정을 meta 수로 하면** 이관 중 셀이 완료로 잡힌다 | pkl 수(=이관 완료)로 판정 — 수집 인덱서·중추 셀표·내 러너 3층 모두 적용됨 |
| 6 | **원격 장시간 작업을 일반 bg 로 돌리면 harness 가 죽인다** | `setsid nohup … &` 필수 |
| 7 | **v6 지터 열이 둘** — `jitter_idx`(좌표) vs `jitter_reset_idx`(성분). oven·washer 는 후자가 전부 0이라 좌표로 쓰면 (scene,noise)당 5판이 한 키로 뭉친다 | 추출기가 `jitter_idx` 우선(수정 반영됨) |
| 8 | 공유 HDD I/O 경합 중 **일시적** `Bad CRC-32` — 파일은 무손상(직후 다시 읽으면 정상) | 재시도 3회, 소진 시 실패(조용히 삼키지 않음) |
| 8b | **지속성 CRC 손상** — 재시도해도 계속 실패하면 그건 진짜 손상이다. 실측: `segA_scene/OpenDrawer_right__s1.npz` 가 09-07 손상(중추 세션 발견). 8 과 구분하는 법 = **재시도 후에도 같은 멤버에서 실패**하는가 | §6b 복구 절차 |
| 9 | **scene shard 를 `segA/` 에 두면** ae_cluster 가 별개 instruction 으로 잡는다 | `segA_scene/` 분리 |
| 10 | 병합 시 **phase 코드북 재매핑 누락** → 에러 없이 phase 라벨이 섞인다(코드북은 shard 마다 독립 생성) | `merge_scene_shards.py` 가 union+remap (합성 데이터로 검증됨) |
| 11 | **모니터가 SSH 연결을 계속 점유** — persistent `tail -F` 가 하나씩 물고 있어 연결 한도에 걸린다 | 작업 끝나면 즉시 정지. 원격 sshd MaxStartups 10 |

### 6b. 손상 shard 복구 (지속성 CRC)

scene shard 하나가 깨져도 **그 키의 병합본이 멀쩡하면 재추출 없이 되살릴 수 있다** —
병합본은 scene shard 를 이어 붙인 것이라 `scene` 열로 잘라내면 원본과 같은 행이 나온다.

```python
# 병합본에서 해당 scene 행만 잘라 scene shard 재생성
import numpy as np
with np.load(MERGED, allow_pickle=False) as z:          # segA/<slug>.npz
    m = z["scene"] == SCENE
    out = {k: (z[k][m] if z[k].shape[:1] == z["scene"].shape else z[k]) for k in z.files}
# ep_id 는 0부터 다시 매기고, meta_json 의 n_episodes·sigs 도 그 scene 것만 남길 것
```

**전제와 확인 사항**
- 병합본이 **손상 이전에** 만들어졌고 그때 감사를 통과했어야 한다. 병합은 scene shard 를
  읽어 만들므로, 손상이 병합보다 앞섰다면 병합본도 같은 손상을 물려받는다.
- 복구 후 **판수·succ·j 분포를 원래 감사 기록과 대조**하라(`audit_cells_scene.tsv` 의 그 행).
- 손상본은 지우지 말고 `.bad_crc_<날짜>` 로 보관 — 원인 추적(디스크·이관·동시 쓰기)이
  남아 있다. 삭제는 사용자 지시 후.
- 09-07 사례는 중추 세션이 이 방법으로 `OpenDrawer_right__s1`(50판)을 복구했고 CRC 검증까지
  마쳤다.

---

## 7. 협업 인터페이스 (누가 무엇을 기대하나)

| 세션 | 내게서 받는 것 | 비고 |
|---|---|---|
| **연산자 설계** | 번들·labels 경로 + `per_cluster.<c>.exclude` | ck8 fit 대상 선정에 사용. **번들 파생 라벨 == labels 전행 일치**를 그쪽이 검증(불일치 시 중단) |
| **fail detector** | shard 경로 + 셀 감사 요약 | **파일명 vs meta instruction 불일치 시 fail-loud** 가드 있음(내 rename 결함을 이게 잡았다) |
| **전체 파이프라인(중추)** | 추출 완료 통지 + 감사표 | 셀표(`v6_loko_cells.tsv`)와 3중 대조(인덱스 success · 아카이브 pkl · 내 shard succ) |
| **데이터 추가 수집** | (받는 쪽) pkl 50/50 완료 통지 | 인덱서 SR 열 포함. 통지 오면 내가 바로 추출(중추 요청 불요, 합의됨) |

**통지 형식** (그대로 쓰면 된다): 경로 · 판수/record/succ · 셀 분포(j별) · 감사 통과 여부 ·
주의사항(재추출본이면 명시).

---

## 8. 지금 열려 있는 것

1. **승준 SSH 복구 확인** — 복구되면:
   `pgrep -af '[c]hain|[a]e_cluster|[e]xtract_grid'` (0이어야 함) ·
   잔여 `*.npz.tmp`·`*.metatmp` 정리 · `du -sh analysis/grid_phase_v6/*` ·
   `df -h ~/datasets`. 급하지 않다.
2. **소비 세션 산출 대기** — 연산자 ck8 fit, detector 재학습. **내 쪽 대기 작업은 없다.**
3. **다음 라운드 후보** (내 판정에서 파생, 착수 지시는 없음):
   - **rack 계열 cluster 의 scene 성분 분리** — 지금은 "잔차화 시 margin 절반 붕괴"까지만
     확인. scene 부분공간을 빼고 cluster 를 다시 만들면 phase 다움이 오르는지가 열린 질문.
   - **PPCC 첫 record 신호를 후보 단위로 재측정** — bread 의 j 잔차화 0:1 AUROC .698 은
     에피소드 단위 추정이다. rsN_llr 채점기는 같은 시점 N 후보를 가르므로, 후보 단위
     재현이 되면 그게 직접 증거가 된다.
   - **창별 등록 게이트** — 초기 절대 창(0:1/0:3/0:10)은 편향이 없으니 그대로, 중·후반은
     phase dwell 고정 예산으로 바꿔 생존 판 편향을 없앨 것.
