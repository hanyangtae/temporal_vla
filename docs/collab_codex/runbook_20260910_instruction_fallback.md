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
