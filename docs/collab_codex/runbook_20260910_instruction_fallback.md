# Scene별 phase 통합 연산자와 k8 fallback 비교

사용자 확정: instruction 단위는 **scene별 기존 학습 풀에서 k8 구분만 제거**한다. scene 간 데이터는 합치지 않는다.

- 기존 완료 23scene/9instruction, 동일 target j의 n0..9. 2arm 총460판.
- instruction_b08: detector 발화 시 scene 공통 plain setM.
- cluster_instruction_b08: detector 발화 시 기존 k8 plain setM, 해당 cluster 미등록이면 동일 scene 공통 setM.
- 두 arm 모두 seed1 그대로 재실행. reseed fallback 없음. GT phase, base replay 없음.
- β=.8, L12만, fit denoise index3/all49 token mean; 적용 모든 denoise call/all49 동일 delta(common shift v2).
- detector는 기존 target 외40판 학습 checkpoint 그대로. 연산자는 기존50판 중 target 성공만 제외, target 실패 허용.
- 양 arm의 공통 연산자는 **길이 조정 없이 모든 적격 record**를 사용하고 성공/실패별 record mean을 계산한다. truncation, dwell cap, episode reweighting 모두 없다(사용자 최신 지시). k8 연산자 NPZ는 바이트 그대로 복사한다.
- `configs/experiments/instruction_fallback_20260910/episodes.tsv`: 기존 검증 완료 target230판; `plans.json`: 기존 replay plan 보존(구/신 좌우 치환 금지).
- 원격 archive의 기존 prepared NPZ만 읽는다. 원본/기존 fit/eval은 변경하지 않는다.

학습:
```bash
python scripts/analysis/grid_phase/fit_instruction_fallback.py \
 --manifest configs/experiments/instruction_fallback_20260910/fit_manifest.json \
 --out outputs/analysis/grid_phase/instruction_fallback_build_20260910
```

새 output prefix: `outputs/eval/robocasa/groot_n15/og_instruction_fallback_20260910`.
GPU는 발사 시 docs/05 및 공통 하네스를 확인한다. 첫 inference smoke에서 routing/seed/L12/common shift 및 fallback 로그를 검증한 뒤 계속한다. 완료 조건은 각 arm230판 및 sidecar/contract 검사 통과이다.

## 실행 기록 (2026-09-10 06:09 UTC)

- fit 코드870b142, archive fit 완료23/23. 새 NPZ·metadata 총약1.8MB 회수, detector 재fit 없음.
- 로컬 중앙 `launch_instruction_eval.py`가 실제 worker hostname/빈 GPU를 확인하고 common checkout의 `groot_harness.py broker` reserve/verify를 호출한다. 원격별 원장이나 reverse SSH credentials는 만들지 않는다. 각 queue는 기존 with_gpu_lease.sh와 같은 세션명을 쓴다. queue 종료 시 GPU가 비었을 때만 receipt를 반납한다.
- kanu: GPU5,6,7 ×2, wrapper1662613, queue1662768, 16jobs160판.
- srv48(worker1): GPU2 ×6, local wrapper1662614, remote queue2347625, 14jobs140판.
- srv50(worker2): GPU1 ×6, local wrapper1662615, remote queue373430, 16jobs160판.
- 각 머신 port9800부터. 실제 세 머신 `/act_with_features` HTTP200 확인.
- 중앙 로그: `outputs/analysis/instruction_fallback_20260910/{kanu,worker1,worker2}.log`; receipt는 같은 디렉토리의 `*_reservation.json`.
- 결과는 output prefix 아래 머신별 하위 폴더에 저장한다. 구 키로 실행한 원본 scene은 `artifact_instruction`으로 신 키 결과 보고에 정규화하며 재현 plan 자체는 수정하지 않는다.

검증: host14 routing/contract tests, lerobot container의 routing4 + 기존 dwell3 테스트 및 runner 테스트 통과. kanu16jobs dry-run 통과. 23scene 모두 길이조정 none/scene 1개 source, 두 arm 공통 NPZ 동일, 기존 k8 NPZ 바이트 동일 확인. 첫 episode sidecar가 저장되면 same seed 및 fallback reason의 실행값을 추가 확인한다.
