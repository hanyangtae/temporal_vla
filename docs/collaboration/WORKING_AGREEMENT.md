# 김상우 · Detector 개발 시 주의사항

버전: 0.2 / 2026-09-18. 대상: detector 개발 담당 김상우와 함께 작업하는 AI.
이 문서의 목적은 김상우가 이전 채팅 없이 detector 개발에 필요한 입력·데이터·평가 조건과 주의사항을 이해하는 것이다.
`확정`은 사용자가 정한 연구 조건이고 `제안`은 아직 동료와 통합 검증하지 않은 구현 방식이다.
문서가 있다고 학습·수집·GPU 발사 권한이 새로 생기는 것은 아니다. 담당자가 맡긴 작업 범위 안에서 진행한다.

## 0. 용어정의
- steering hook: activation에 연산자를 적용하도록 끼운 코드
- jitter(j): 한 robocasa scene내에서 물건, 로봇 base의 위치를 약간 변경한 variation
- diffusion noise(n): 한 instruction, scene, jitter 내에서 diffusion noise를 다르게 해서 모으는 data

## 1. Detector 개발의 목적과 범위

김상우는 task_classification에서 detector를 개발한다. 목표는 **오발화를 줄이면서 실패 감지를 유지하고,
실패 사건 전 발화를 늘리는 것**이다. 사건 후 stuck 상태를 감지하는 능력도 별도로 본다.
모델 구조는 자유롭게 선택할 수 있으며 LSTM이나 두 MLP로 고정하지 않는다.

박경태는 steering과 전체 파이프라인 통합을 담당한다. Detector 개발에 필요한 접점은
activation 입력, 현재 inference step의 개입 필요 여부, action 실행 전 판정 시점이다.
Steering 연산자 설계·beta·fallback arm·GR00T 수집 운영은 이 문서의 개발 과제가 아니다.
개입 후 detector 상태 정렬은 양쪽이 나중에 결정할 문제로 남긴다.

## 2. 확정된 Detector 규칙

| 항목 | 지켜야 할 계약 |
|---|---|
| 입력 | activation 기반. 제공 가능한 최대 범위를 공유하고 모델별 부분집합·전처리를 선언한다. 모델 구조를 LSTM이나 MLP로 고정하지 않는다. |
| 시간 | inference step 기준.|
| 출력 | 현재 inference step의 개입 필요 여부. 해당 action이 환경에서 실행되기 전에 판정 완료. |
| 인과성 | 현재 시점에 아직 얻을 수 없는 activation·미래 결과·최종 성공 라벨을 온라인 입력으로 쓰지 않는다. |
| 데이터 버전 | 바뀌는 데이터 pool과 실험용 frozen manifest를 분리한다. 학습 중 snapshot을 교체하지 않는다. |
| 실행 식별 | scene/j/n·seed·성공 여부만으로 동일 실행/영상으로 조인하지 않는다. 아래 실행 식별 조건을 모두 기록·대조한다. 동일 영상 파일은 전체 SHA256으로 확인하며, 재실행은 같은 조건이어도 별개 rollout이다. |
| 비교 | detector 판별 성능과 steering의 SR·구제·파괴를 구분한다. 다른 머신 참고 영상의 라벨을 원본과 동일시하지 않는다. |
| 버전 통합 | detector commit과 모델·전처리·threshold를 함께 고정한다. submodule은 검증한 commit만 갱신한다. |

### 실행 식별: 같은 영상과 같은 재현 조건의 구분

- **원본 식별:** rollout_id/run ID, 실제 생성 machine, 원본 plan ID·hash, rollout·video·activation 각각의 전체 SHA256 및 생성·파생 관계. `pkl_sha256`이나 짧은 `sig`만으로 영상 동일성을 판정하지 않는다.
- **환경 좌표:** task/env와 원문 instruction·좌우 키 매핑, scene_idx, 실제 layout/style, 주방 후보 목록·순서, fixture/객체 asset, jitter_idx와 실제 reset_idx·lat/back·좌표계·부호를 보존한다.
- **초기 상태·난수:** ep_meta와 주입 시점, 초기 simulator/robot/객체 상태, reset 순서, env/inference seed, noise_idx→seed 매핑, RNG 초기화·소비 순서, 실제 policy noise 및 reseed 이력을 대조한다.
- **정책·실행 환경:** checkpoint·전처리·normalization hash, 코드와 submodule commit·미커밋 patch, 실제 모듈 경로, container/라이브러리, GPU·driver·renderer/backend/vendor·dtype·결정성 설정을 대조한다.
- **관측·action·개입:** camera·해상도·pose와 영상 전처리, state/history 입력, horizon·denoise·실행 action 수, action 후처리·controller·physics timestep, 종료 조건, detector/operator·threshold·fallback·발화 및 hidden-state 처리를 대조한다.
- **영상·activation 대응:** render 주기·시작 offset·누락/중복 frame·전체 frame 수·PTS/FPS, codec·pixel format·crop/overlay, activation hook·축·pooling·dtype·후보/개입 pass를 기록한다. 라벨은 실제로 본 video hash와 frame↔env_step↔inference_step 대응에 연결한다.

**설정 일치는 동일 영상의 충분조건이 아니다.** 같은 파일은 전체 video SHA256 일치로 확인한다.
파일 hash가 다른 경우 같은 프레임열을 주장하려면 동일 디코딩 규약의 모든 픽셀·프레임 수·표시 시각을 확인해야 한다.
재인코딩·편집본은 원본 hash와 변환 대응을 가진 파생 영상이고, 같은 seed의 재수집은 별개 실행이다.
원본 영상의 byte-copy는 저장 머신이 달라도 같은 파일이다. 누락된 조건은 `미확인`으로 기록한다.
실제 재실행 비교에는 초기 상태와 step별 관측·noise·실행 action·state 검증도 필요하며,
같은 결과나 비슷한 영상만으로 원본의 사건 라벨을 옮기지 않는다.
전체 항목·판정 수준은 [데이터 계약의 재실행 조건 대조표](data_contract.md#재실행-조건-대조표)를 따른다.

## 3. Detector 학습·평가에서 주의할 점

### 데이터 분할과 길이 통제

- 같은 rollout의 frame/activation을 train과 test에 나눠 넣지 않는다. 복사본·파생 artifact도 원본 ID 기준으로 함께 분리한다.
- 기존 target-j 기준 비교에서는 같은 instruction/scene의 target j 10판 전체를 제외하고 다른 4j 40판으로 detector를 학습한다.
  Target의 실패판도 detector fit에 포함하지 않는다. 기존 operator는 target 실패를 쓰기도 하지만 그 학습 pool을 detector에 재사용하면 안 된다.
- 위 40판 전체에는 성공·실패가 필요하다. 각 j가 개별적으로 양 클래스를 가져야 한다는 추가 조건은 없다.
- 새 split이나 학습 방법은 별도 실험으로 명시할 수 있다. target-j heldout과 scene heldout의 일반화 범위를 구분한다.
- 표준화·PCA·AE·cluster 등 학습하는 전처리도 해당 split의 학습 데이터로 fit한다.
  기존 pretrained artifact를 사용하면 학습 데이터 중첩 여부와 고정/재학습 여부를 밝혀 평가 누출을 확인한다.
- 모델 선택·threshold 조정·calibration에 사용한 데이터를 최종 평가 데이터와 구분한다.
- 성공/실패의 길이 차이, 긴 episode의 과도한 기여, scene/jitter별 불균형을 확인한다.
  기존 k8 조건과 비교할 때는 cluster별 길이 통제 방식과 episode별 기여량을 명시한다.
  다른 방식은 사용할 수 있지만 기존과 같은 통제를 했다고 추정해서 쓰지 않는다.
- 실패 rollout의 후반 phase/cluster 부재, 정상 timeout·행동 실패는 손상이 아니다.
  표본이 적으면 확인 불가를 표시하고 임의 cutoff로 유리한 표본만 남기지 않는다.

### 예측 대상과 시간

- Episode의 최종 성공/실패 라벨과 특정 시점의 실패 사건 라벨은 다르다.
  최종 실패 라벨을 모든 frame의 '현재 실패 상태' 정답으로 해석하지 않는다.
- 최종 성공한 rollout에도 실패 사건과 회복이 있을 수 있다.
  성공판 발화율 전체와 검토된 무사건 성공판의 발화율을 분리해서 본다.
- Sequence 모델뿐 아니라 pooling·window·정규화에도 미래 record가 섞이지 않게 한다.
  양방향 window, 전체 episode 평균, 종료 후에만 아는 최종 길이 등은 online 입력이 될 수 없다.
- 같은 inference_step의 denoise slot과 서로 다른 시간의 inference_step을 혼동하지 않는다.
  사건 frame/env_step을 inference_step으로 바꾸는 offset과 대응은 데이터별로 확인한다.
- 사건 전 감지 성능은 사건 전 라벨과 발화 시점으로 직접 평가한다.
  단순 성공/실패 분리 점수만으로 사전 감지 능력을 주장하지 않는다.

### 보고할 성능

- 성공판 전체 발화율, 검토된 무사건 성공판의 발화율, 실패판 발화율·미발화 수를 분자/분모와 함께 본다.
- 첫 발화가 첫 실패 사건 전인지 후인지와 시차를 보고한다. 여러 사건은 보존하고 어느 사건과 비교했는지 명시한다.
- 사건 후 stuck 감지는 사전 예측과 구분한다. 미검토·라벨 누락은 별도 분모로 드러낸다.
- Frame 지표와 rollout 지표를 구분하고 instruction/scene별 편차를 함께 확인한다.
- Score는 모델별 의미가 다르다. 두 MLP의 점수가 독립 출력이라면 합이 1인 성공/실패 확률로 가정하지 않는다.
- Detector의 발화 개선과 steering의 구제/파괴는 별도 평가다. 구제 실패만으로 detector 오발화라고 판정하지 않는다.
- 현재 inference step의 판정이 action 실행 전에 끝나는지, 실제 실행 환경의 지연과 episode reset을 확인한다.

## 4. 아직 결정하지 않은 사항

| 미결 또는 누락 | 처리 |
|---|---|
| 공유 저장소 URI/권한, frozen manifest 없음 | 데이터 의존 학습·평가는 보류. 로더·인터페이스·합성 입력 검증은 가능. |
| 전달 데이터별 activation 범위 | DiT 최대 16layer×4denoise×49token×1536, 확인 저장본 7layer 범위는 detector_contract.md 참조. 전달 묶음 전체의 스키마와 VL token 추가 제공은 별도 확인. |
| detector 개발 branch/commit | 후보를 명시하고 호환성 확인. 임의로 main 최신 commit에 맞추지 않는다. |
| 개입 후 상태 정렬 | LSTM의 현행 방식을 전체 모델의 계약으로 확정하지 않는다. offline detector 개발은 진행 가능. 새로운 online 조합은 결정 전 통합 완료로 표시하지 않는다. |
| 지연 상한 | 지연을 측정해 보고. 임의 숫자로 합격 판정을 만들지 않는다. |
| 사건 라벨 없음 | 해당 snapshot의 검토 상태 확인. 기존 성공판 미라벨=무사건 합의를 새 미검토 데이터에 확장하지 않는다. |

## 5. 파이프라인에 연결할 때만 필요한 사항

입력 선택·pooling·전처리·threshold와 모델 버전을 명시해 통합 담당과 공유한다.
개입 후 hidden state 처리와 재-forward의 시간 처리는 아직 미결이며 임의로 기존 LSTM 규칙을 일반화하지 않는다.
Detector 개발만을 위해 원본 baseline을 자동 재수집하거나 steering hook을 변경하지 않는다.
추가 activation capture 또는 GR00T 실행이 필요하면 통합 담당과 범위를 맞춘다.
GR00T 운영 규칙은 temporal_vla의 `docs/05_gpu_server_rules.md`와 공통 하네스에 있으며,
그 자원 수치를 detector 학습에 자동 적용하지 않는다.

## 6. GitHub 연결 방식

Detector 코드 개발은 task_classification에서 진행하고, temporal_vla는 검증한 detector commit을
submodule로 고정하는 방식을 사용한다. 모델 내부 변경과 파이프라인 통합 변경을 구분한다.
전처리·모델·threshold·입력 정의가 같은 버전을 가리켜야 한다.
구체적인 후속 PR 양식, 작업 관리, AI 간 인계 방식은 이번 최초 전달 자료에서 정하지 않는다.
