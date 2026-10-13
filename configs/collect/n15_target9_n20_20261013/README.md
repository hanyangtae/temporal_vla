# 9개 target scene noise 확장 (2026-10-13)

사용자 표의 scene/jitter split을 보존하고 각 j에 n10~19를 추가한다. 기존 train40/test10 → 완료 후 train80/test20. 성공/실패는 미리 정하지 않는다.

| Instruction | scene | test j | 원래 머신 | 추가 |
|---|---:|---:|---|---:|
| DishwasherRack/out-left | 3 | 2 | worker1/srv48 | 50 |
| DishwasherRack/out-right | 1 | 2 | worker1/srv48 | 50 |
| OpenDrawer/left | 2 | 2 | kanu | 50 |
| OpenDrawer/right | 1 | 4 | kanu | 50 |
| OvenRack/out-left | 2 | 4 | worker1/srv48 | 50 |
| PPCC/bread | 4 | 1 | worker2/srv50 | 50 |
| PPCC/candle | 0 | 1 | worker2/srv50 | 50 |
| PPCC/jug | 1 | 1 | worker2/srv50 | 50 |
| PPCC/marshmallow | 4 | 2 | worker2/srv50 | 50 |

新 plan 748863ce9233은 부모 superset 4ab360df2b71의 noise_seeds만 [1300000..1300019]로 확장한다. 모든 원본 instruction/scene/jitter/캡처/환경 정의는 동일하다. source_plan은 requests.tsv에 보존한다. n0~9는 이전 plan 아래에 그대로 있으며 새 plan 아래 중복 수집하지 않는다. 후속 분석에서 source_plan+machine+instruction+s+j+n 계약으로 합쳐야 한다.

machine별 skip 파일은 기존 n0~9 및 이번 대상 밖 scene의 n10~19를 제외한다. 이는 해당 제외 셀이 새로 수집됐다는 증거가 아니다. 요청된 실제 신규 수집 범위는 requests.tsv 450행이다.

수집 검증은 expansion_mode=noise 계약 사용. 기존 scene 추가 모드의 n10 규칙은 변경하지 않는다. noise 모드는 기존10 seed 보존, 20개 고유 seed, 환경 좌표 불변, 대상 외 판 제외를 검사한다.

실행 상태/로그는 main checkout outputs/collect/target9_n20_20261013/에 있다. 48번은 NVML driver/library mismatch, 50번은 전체 GPU 기존 작업 점유로 초기 발사 보류. 타 머신 이관 금지. kanu는 GPU6 2serve, staging 8GB 상한 및 승준 롤링 shipper를 사용한다. 추가 데이터 생성은 baseline 수집이며 detector 재학습/eval은 이 작업 범위에 없다.
