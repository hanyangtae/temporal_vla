# Target9 n40 수집 (2026-10-25)

기존 9 instruction/scene × 5j의 n0~19는 개발 자료로 유지한다. 새 n20~39만 final test 후보이며 seed는 251025020~251025039이다. 과거 plan의 1300020대 seed 재사용을 피했다. 기존 환경·layout/style·좌우·checkpoint·action(16 predicted/5 executed)·activation 계약은 불변이다. 새 plan 2f26fb046fc4, 부모 748863ce9233.

requests.tsv: 새 test 900판(kanu 200 / srv48 300 / srv50 400). folds.tsv: 5개 heldout_j별 역할. 모든 새 seed는 전 fold의 train/SAFE보정/AEcluster/potential/설정 선택에서 제외. 다른 j의 새 seed도 reserved_not_train이다. fold는 독립 학습/분석하며 최종 test 결과로 선택하지 않는다.

backfill_development.tsv: 승준 원본 대조 결과 srv48 n10~19 누락 108판(Dish L s3 49, Dish R s1 50, Oven L s2 9). 이는 신규 test가 아니다. 기존 n20 계약/원래 머신에서 별도 보충하며 n40 계약으로 원본 셀을 재수집하지 않는다. 기존 정상 완료 파일은 변경하지 않는다.

2026-10-25 preflight: srv48/50 모든 GPU 점유. kanu GPU6 2serve 우선 실행. 승준 여유226GiB, 전체900판+108보충은 보수적432MB/판 기준 약435GB이므로 전체 동시 발사 금지; 우선200판은 약86GB 상한 예상. 새 headless 수집 컨테이너 Docker 로그10m×3, staging8GB backpressure. 전송 후 SHA256 검증을 통과한 경우만 staging 정리.

신규 task 준비 후보(설치된 task 클래스 존재 확인, scene/reset/jitter feasibility 아직 미확정): PickPlaceSinkToCounter(대안DrawerToCounter), OpenCabinet(대안CloseFridge), TurnOnSinkFaucet(대안TurnOnMicrowave), PickPlaceCounterToStove. 아직 신규task rollout 발사하지 않음. Coffee/apple 제외 유지. atomic65 human demo 전체 사용은 별개이며 여기서65종 rollout 수집으로 해석하지 않음. 현재 로컬 cache/outputs 탐색만으로6500판 확보 여부 미확인; 신규 다운로드하지 않음.
