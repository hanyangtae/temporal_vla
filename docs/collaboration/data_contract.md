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

## 무엇을 ‘같은 영상’이라고 부를 것인가

**설정값이 같다는 것만으로 동일 영상이라고 판정하지 않는다.** 다음 세 수준을 구분한다.

| 주장 | 필요한 증거 | 충분하지 않은 근거 |
|---|---|---|
| 같은 영상 파일 | 파일 전체의 SHA256 일치. 파일 크기도 함께 기록 | 같은 파일명·경로·scene/j/n·성공 여부 |
| 같은 프레임열 | 같은 디코딩 규약에서 프레임 수·해상도·픽셀 배열과 표시 시각(PTS/time base)이 모두 일치 | 영상 파일 hash가 다른데 육안으로 비슷함, FPS 숫자만 같음 |
| 같은 rollout에서 만든 파생 영상 | 원본 rollout/video hash와 생성 이력, 원본 frame/env-step 대응표 | 같은 seed로 재수집, 같은 머신에서 재실행 |

재인코딩·자르기·속도 변경·자막 삽입 영상은 별도 artifact ID를 갖는다.
원본 이력이 확인돼도 픽셀/타임라인이 달라지면 ‘동일 영상’ 대신 ‘같은 rollout의 파생 영상’으로 표시한다.
특히 lossy 재인코딩은 같은 장면이어도 픽셀 hash가 달라진다. 오디오가 포함된 파일의 전체 동일성은
파일 hash로 판정하고, 프레임열 비교만 했다면 오디오까지 같다고 주장하지 않는다.

`rollout.pkl`의 hash와 `video.mp4`의 hash는 별개다. 기존 meta의 `sig`/`pkl_sha256`만으로
영상 동일성이 입증됐다고 해석하지 않는다. 짧은 sig는 조회 보조로 쓰고 전체 hash를 보존한다.
같은 영상 파일을 다른 저장소로 복사한 경우 저장 위치나 복사한 머신이 달라도 같은 artifact다.
여기서 보존할 machine은 복사 위치가 아니라 **실제 생성한 머신**이다.

## 재실행 조건 대조표

아래는 재현 후보를 대조하기 위해 필요한 조건 목록이다. 값이 같아도 GPU·물리 시뮬레이터 등의
비결정성 때문에 동일 궤적/영상이 보장되지는 않는다. 실행 이력과 실제 출력 검증까지 필요하다.
기존 artifact에 없는 항목은 `unknown`으로 기록하며 일치한 것으로 처리하지 않는다.
적용되지 않는 항목은 근거와 함께 `not_applicable`로 구분한다.

| 범주 | 반드시 대조·보존할 항목 |
|---|---|
| 원본·버전 | dataset snapshot/manifest hash, 원본 rollout_id/run ID, 수집 시각, rollout·video·activation 각각의 전체 hash, 파일 크기, 원본/재실행/파생 관계 |
| Task·언어 | env/task class와 ID, 원문 instruction, canonical 키, 좌우 키 mapping 버전. 동일 문자열 key라도 구/신 좌우 의미가 다를 수 있음 |
| Scene·plan | plan ID와 내용 hash, scene_idx, 실제 layout_id/style_id, 주방 후보 목록과 순서, fixture/객체 종류·asset 버전·언어와 target 매핑. layer_id 표기가 있으면 실제 필드가 layout인지 먼저 확인 |
| Reset·지터 | jitter_idx와 실제 reset_idx, lat/back 등 오프셋 값·단위·좌표계·부호, base pose, reset 재시도/호출 순서, ep_meta 사전 주입 여부 및 주입 시점 |
| 초기 물리 상태 | ep_meta, 모델/XML 및 초기 simulator state, robot base·관절 qpos/qvel, gripper, 객체/fixture pose·관절 상태. 저장된 초기 state hash와 reset 직후 관측 hash |
| 난수 | env/scenario seed, inference/policy seed, noise_idx→seed mapping, Python/NumPy/Torch/CUDA 및 simulator RNG의 초기화·소비 순서, episode 간 reset 규칙, 실제 policy noise 또는 그 hash, reseed offset·seed·발생 step |
| 정책 | 모델 계열, checkpoint 파일 hash, config·normalization stats·processor/tokenizer 버전/hash, embodiment, action horizon, denoise 수·schedule, precision, eval mode 및 sampling 옵션 |
| 실행 코드·환경 | 수집기·serve·policy·benchmark·모든 관련 submodule commit과 미커밋 patch hash, 실제 로드한 모듈 경로, container image digest/환경 lock, Python/PyTorch/CUDA/cuDNN/MuJoCo/RoboCasa 등 버전 |
| 하드웨어·수치 실행 | 수집/serve/render machine, GPU 모델, driver, renderer backend(EGL/OSMesa等)와 vendor(NVIDIA/Mesa等), 렌더 device, dtype/TF32·deterministic 설정, batch 크기·동시 실행·thread 설정. 같은 hostname만으로 같다고 판정하지 않음 |
| 모델 관측 | camera 이름·순서·해상도·intrinsics/extrinsics, state 항목·순서·단위, resize/crop/flip/rotate·색상 변환·normalization, 관측 history 및 추론 시점. raw observation과 model input hash를 구분 |
| Action·물리 진행 | 예측 chunk 길이와 실제 실행 action 수, queue/reset 규칙, action 순서·단위·정규화/역정규화·좌표계·rotation/gripper 처리·clipping, control rate·physics timestep·substep·controller 설정, settling/warmup, horizon·success/termination/timeout 조건 |
| 개입 arm | baseline/reseed/operator 구분, detector·threshold·전처리·operator hash, beta·layer·denoise·token 범위, gate/cooldown/fallback, 실제 발화·개입·재-forward 순서, 개입 후 hidden state 처리 |
| Activation 추출 | 온라인 동시 capture인지 사후 재실행인지, hook 위치(pre/post)·layer/denoise/token 범위·pooling·dtype, inference record ID, 후보/steered pass 구분. 새 추출 artifact를 기존 영상에 연결하려면 원본 실행 lineage 확인 |
| 영상 생성 | 실제 render 카메라·배치·해상도, record 시작 offset, env step 대비 render 주기, warmup 포함 여부, frame drop/duplicate, 총 frame 수·PTS/time base·FPS, encode codec·설정·pixel format, crop/overlay/자막·후처리 버전 |

초기 state나 renderer 정보가 없는 과거 데이터는 이 표를 사후 추정해 채우지 않는다.
설정 누락은 ‘재현 계약 불완전’으로 보고하며, 원본 영상의 정확한 복사본 사용 자체를 막는 사유는 아니다.

## 실제 동일성 확인과 라벨 연결

1. **복사본 확인:** 제공 영상의 전체 hash를 원본 manifest와 비교한다. URI만 변경된 복사본은
   동일 artifact로 연결하고, 다른 hash의 파일을 기존 경로에 덮어쓰지 않는다.
2. **재실행 확인:** 새로운 rollout_id를 부여한다. 위 설정 대조에 더해 초기 state/관측,
   inference별 입력·noise, env-step별 실행 action·state, 종료 step 및 결과를 비교한다.
   전부 exact-match인지 허용오차 비교인지 구분하고, 허용오차 비교를 동일 영상이라고 부르지 않는다.
   값이 같은 경우에도 같은 실행이었다고 주장하지 않고 ‘별도 실행, 비교 구간 일치’로 기록한다.
3. **렌더 확인:** 궤적이 같아도 카메라/renderer 차이로 영상은 다를 수 있다.
   동일 영상 주장은 파일 hash 또는 위의 픽셀·PTS 전수 비교로 별도 확인한다.
4. **라벨 연결:** 라벨에는 실제로 사람이 본 video hash와 rollout_id, label snapshot을 연결한다.
   video frame↔env_step↔inference_step 및 그 offset/주기/누락 처리를 기록한다.
   파생 영상으로 라벨을 옮길 때는 원본 frame 대응과 변환 이력을 남긴다.
   재수집 영상 라벨은 좌표가 같다는 이유만으로 원본에 이식하지 않는다.
5. **Steering 비교:** baseline과 steering은 개입 후 궤적이 달라지는 실험이다.
   동일 영상 검증과 구분하여 최초 개입 전의 입력·noise·action·state 대응을 확인한다.
   개입 전 불일치나 참고 base의 성공/실패 반전은 구제·파괴 집계에 숨기지 않는다.

예: `bread s4 j3 n2`가 같아도 kanu 원본과 worker2 재수집본은 별개 실행이다.
최종 성공 여부가 우연히 같아도 라벨·발화 시점·구제 결과를 교차 사용하지 않는다.
반대로 kanu 원본 MP4를 worker2에 byte-copy해서 전체 hash가 같다면 같은 영상 파일이다.

이 목록은 **협업 시 필요한 manifest 항목과 확인 기준**이다. 현재 모든 수집본에 이 필드가
존재하거나 자동 검사가 구현돼 있다는 뜻은 아니다. 최초 전달 시 확보된 증거와 미확인 항목을 구분한다.

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
