# 추가 수집본 중 미완료 세 scene 평가

PPCC bread s4 j3, marshmallow s3 j4/s4 j2는 plan `4fa6496cd684`의 기존 150회 완료 결과와 계약·seed·결과행을 대조했으며 재실행하지 않는다.

| 신 키 | target | S/F | other40 S/F | plan | 수집/평가 머신 |
|---|---|---|---|---|---|
| OpenDrawer/right s3 | j1 | 9/1 | 38/2 | d8c7e569aa37 | worker2 / srv50 |
| DishwasherRack/out-left s3 | j0 | 8/2 | 19/21 | 4ab360df2b71 | worker1 / srv48 |
| DishwasherRack/out-right s4 | j0 | 1/9 | 3/37 | 4ab360df2b71 | worker1 / srv48 |

마지막 scene의 추천 j4는 성공2/실패8이나 other40 성공이2개뿐이다. 기존 `loo_cp_band`는 성공3개 미만이면 반환하지 않고, 두 성공의 LOO는 각 fold의 ddof=1 표준편차가 정의되지 않는다. 성공1개인 j0/j1 중 j0를 선택해 기존 계산을 유지한다. 이는 target 성공 최소치를3으로 추가한 것이 아니다. 성공3개 calibration도 희소하므로 nominal coverage를 주장하지 않는다.

원본 인덱스: `outputs/analysis/topup_remaining_eval_index_20260910/rollouts.tsv`. 세 scene 모두 50개 pkl 존재. 신규 eval 중복0 확인. 원본 SHA 및 meta/seed/label은 `prepare_ck8_raw_scene.py`가 원격에서 검증한다.

설정: `configs/experiments/v6_ck8_remaining_20260910/{srv50_drawer,srv48_dish}`. 각 source_index는 전체50 또는100판, episodes는 target10 또는20판만 포함한다. 모델 k8/L12/fit denoise3, hook all denoise/common shift v2, target 제외 detector40, target 성공만 제외 operator, beta plain .8/jfair .9, missing-cluster reseed fallback을 유지한다. GT와 base replay는 실행하지 않는다.

## 실행

CPU repo `/home/kimseungjun/workspace/temporal_vla_ck8_dwell` 및 두 GPU 서버에 git으로 코드를 동기화한다. remote helper로 `build_ck8_remaining_20260910.sh`를 실행했다. Drawer 먼저 준비하고 이후 두 Dishwasher를 순차 준비한다.

```bash
REMOTE_REPO=/home/kimseungjun/workspace/temporal_vla_ck8_dwell \
  bash scripts/utils/remote_compute.sh run-bg ck8_remaining_20260910 \
  bash scripts/analysis/grid_phase/build_ck8_remaining_20260910.sh
```

dispatcher는 `--config-dir configs/experiments/v6_ck8_remaining_20260910/<group>`, `--build-dir outputs/analysis/grid_phase/v6_ck8_remaining_build_20260910/<group>`, `--state-dir outputs/analysis/v6_ck8_remaining_dispatch_20260910/<group>`, `--out-root outputs/eval/robocasa/groot_n15/og_v6_ck8_remaining_20260910/<group>`를 쓴다. detached process로 발사하며 live pid는 state 폴더가 정본이다.

- srv50 GPU1, port9760부터, Drawer 5 arm ×10=50회.
- srv48 GPU2, port9720부터, Dishwasher 2scene×5 arm×10=100회; GPU당6모델 한도에서 arm 통합 queue.
- detector만 먼저 준비되면 reseed 먼저. 둘 다 준비되면 통합 queue. 각 stage의 모든 대상 모델이 준비됐는지 확인한다.
- 이전 run의 artifact pull 실패로 부모 상태가 stale이 된 문제를 수정: 전송 오류는 active 평가를 보존하고 재시도, release complete 이후 반복전송 생략.
- custom plan은 git에 동봉하며 원격 repo 상대경로로 전달한다. 새 수집본의 fixture-side 규칙과 같도록 collector의 좌우 부호를 수집 코드와 일치시켰다.

## 검증

- 세 target manifest가 각각 정확히10판이며 plan env/noise seed와 jitter 계약 교차검증 통과.
- 기존 evaluator/ck8/preparation 관련12 tests 통과. collector tests 추가 실행 포함 총21 passed, 4 failed: 기존 fake_predict_with_features가 extra_payload 인자를 받지 못하는 mock signature 문제이며 이번 변경 함수와 별개. 해당 테스트 실패를 성공으로 표기하지 않는다.
- 수집 collector와 평가 collector 파일을 동일하게 맞췄고, 변경 전후 diff는 `_v6_apply_jitter`의 fixture-side 방향 설명·부호뿐이다.
- Python/Bash syntax 및 diff check 통과. 실제 remote serve health와 최초episode는 실행 후 확인한다.

원본·수집 프로세스·타 실험 산출물은 변경하지 않는다. 신규 scope 총150회. 완료 PPCC와 합친 신규 수집 유효6scene은 총300회이지만 이번 추가발사는150회뿐이다.
