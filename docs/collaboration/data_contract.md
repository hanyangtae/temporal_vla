# Rollout 데이터·라벨 공유 초안

## 변경되는 데이터와 고정된 실험

공유 저장소의 데이터 pool은 늘거나 줄 수 있다. 각 실험은 그 시점의 고정 manifest를 사용한다.
데이터 추가·제외나 라벨 수정 시 새 snapshot을 만들고, 과거 실험의 manifest를 덮어쓰지 않는다.
이 규약은 합의 초안이며 자동 snapshot 도구가 구현됐다는 뜻은 아니다.

| 식별 단위 | 필요한 정보 |
|---|---|
| 실행 1회 / rollout_id | 원본 파일 hash·sig, 수집 machine/run, policy·checkpoint·코드 버전 |
| 환경 재현 좌표 | 원본 plan ID, instruction 키와 좌우 의미, scene, layout/style, jitter 정의, noise, env/inference seed, 필요 init state |
| Activation artifact | rollout_id, 파일 URI/hash, 추출 버전, layer/denoise/token 축, dtype/shape, inference↔env step 대응 |
| 라벨 snapshot | rollout_id와 영상 hash, annotator, revision, 사건 목록 전체, 검토 상태, 시간 좌표 |
| 실험 snapshot | 데이터/라벨 snapshot ID, split manifest, 포함·제외 사유, 코드·설정 hash |

같은 instruction/s/j/n이라도 다른 머신·설정·재실행이면 같은 rollout로 취급하지 않는다.
`scene_idx`만으로 주방을 조인하지 않고 원본 plan 및 layout/style을 함께 보존한다.
원본과 재수집 영상의 라벨을 좌표만으로 섞지 않는다.

## 저장소와 공유

- 원본 rollout·영상·activation은 기존 공유 archive에 보관한다. 각 머신의 mount 경로는
  로컬 설정으로 매핑하고, manifest의 논리 artifact ID와 checksum은 같게 유지한다.
- 모델마다 필요한 activation subset을 파생 artifact로 만들 수 있다.
  원본 lineage와 추출 설정을 기록하고 과거 파일을 조용히 교체하지 않는다.
- 작은 manifest와 split은 Git에서 리뷰한다. 대규모 manifest는 공유 저장소에 두고 Git에 URI/hash를 기록한다.
- 실험 manifest가 참조하는 원본을 물리 삭제하면 재현성이 사라진다. 일반적인 실험 제외는
  manifest에서 처리하고, 원본 삭제가 필요하면 영향받는 실험과 보존 여부를 먼저 확인한다.

## 라벨링

기존 공용 labeler를 편집 창구로 쓰고 episode 단위로 작업을 나눈다.
학습·평가에서는 편집 중인 라이브 라벨 대신 고정 snapshot을 읽는다.
사건이 여러 개면 전부 유지하고, episode 최종 성공 여부와 사건 라벨을 별개로 저장한다.

권장 상태는 `unreviewed`, `reviewed_no_event`, `reviewed_events`이다.
이번 기존 성공판에 대해 사용자가 확인한 '미라벨=실패 사건 없음'은 그 snapshot의 규칙으로 기록한다.
앞으로 추가된 미검토 데이터를 자동으로 무사건 라벨로 바꾸지 않는다.
사건 시각은 inference_step, env_step, video frame 간 변환 근거를 함께 저장한다.
고정 FPS나 5 action 실행을 다른 데이터셋에도 일괄 가정하지 않는다.

## 분할과 길이 통제

- 같은 rollout에서 파생한 activation·중복 사본은 split을 넘나들지 않는다.
- 기존 target-j 평가 재현 시 detector는 같은 scene의 target j 전체를 제외한 다른 4j 40판으로 학습한다.
  전체 40판에는 양 클래스가 필요하지만 각 j가 개별적으로 양 클래스를 가질 필요는 없다.
- 연산자는 detector와 다른 fit 규약을 갖는다. 기존 규약에서는 target 성공을 제외하고 target 실패는 허용한다.
- 새 detector 일반화 실험은 별도 split을 명시한다. 기존 target-j 결과와 scene-heldout 결과를 동일하게 부르지 않는다.
- 성공/실패·episode·phase/cluster별 샘플 기여와 길이 통제를 명시한다. 기존 k8 조건을 기준 비교로 보존하되
  detector 구조나 입력 범위 확장은 별도 설정으로 실험할 수 있다.
- Threshold 선택과 calibration에 쓴 데이터는 최종 평가와 구분해 기록한다.

전달 전 검증: 원본 존재·hash·NaN/Inf·shape·episode 경계·record 대응·라벨/seed 정합성.
정상 timeout, 행동 실패, 후반 phase 미도달은 손상 사유가 아니다.
