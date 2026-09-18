# Detector–Steering 최초 전달: 공동 작업 조건

버전: 0.1 / 2026-09-17. 대상: 사용자, detector 담당 동료, 양쪽 AI.
이 문서는 이전 채팅 없이 작업 경계를 이해하기 위한 진입점이다.
`확정`은 사용자가 정한 연구 조건이고 `제안`은 아직 동료와 통합 검증하지 않은 구현 방식이다.
문서가 있다고 학습·수집·GPU 발사 권한이 새로 생기는 것은 아니다. 담당자가 맡긴 작업 범위 안에서 진행한다.

## 0. 용어정의
- steering hook: activation에 연산자를 적용하도록 끼운 코드
- jitter(j): 한 robocasa scene내에서 물건, 로봇 base의 위치를 약간 변경한 variation
- diffusion noise(n): 한 instruction, scene, jitter 내에서 diffusion noise를 다르게 해서 모으는 data

## 1. 목적과 소유권

- 박경태는 steering을, 김상우는 detector를 담당한다. 전체 파이프라인·환경 실행·평가는 temporal_vla 레포에서 관리한다.
- task_classification은 detector 개발 레포이며 temporal_vla의 기존 submodule로 연결한다.
- Detector는 오발화를 줄이고 실패 감지와 사건 전 감지를 개선한다. 사건 이후 stuck 감지는 별도로 진단한다.
- Steering은 detector 판정에 따라 개입하고 실제 실패 구제·성공 파괴를 평가한다.
- Detector 담당은 steering hook·환경 실행을 임의로 변경하지 않는다.
  통합 담당은 detector 학습 조건·threshold·전처리를 동료에게 알리지 않고 바꾸지 않는다.
- 수정 경계 밖의 문제가 보이면 원인·필요 변경을 기록해 해당 담당에게 넘기고, 독립적으로 가능한 작업은 계속한다.

## 2. 확정된 Detector 규칙

| 항목 | 지켜야 할 계약 |
|---|---|
| 입력 | activation 기반. 제공 가능한 최대 범위를 공유하고 모델별 부분집합·전처리를 선언한다. 모델 구조를 LSTM이나 MLP로 고정하지 않는다. |
| 시간 | inference step 기준.|
| 출력 | 현재 inference step의 fire/no-fire. 해당 action이 환경에서 실행되기 전에 판정 완료. |
| 인과성 | 현재 시점에 아직 얻을 수 없는 activation·미래 결과·최종 성공 라벨을 온라인 입력으로 쓰지 않는다. |
| 데이터 버전 | 바뀌는 데이터 pool과 실험용 frozen manifest를 분리한다. 학습 중 snapshot을 교체하지 않는다. |
| 실행 식별 | scene/j/n·seed·성공 여부만으로 동일 실행/영상으로 조인하지 않는다. 아래 실행 식별 조건을 모두 기록·대조한다. 동일 영상 파일은 전체 SHA256으로 확인하며, 재실행은 같은 조건이어도 별개 rollout이다. |
| 라벨 | 최종 성공과 실패 사건은 별도 변수. 모든 사건을 보존하고 라이브 라벨의 고정 snapshot으로 학습·평가한다. |
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

## 3. 기존 steering 평가를 재현할 때의 조건

이 절은 기존 평가 재현 조건이다. 모든 새 detector 연구에 영구 고정된 설계는 아니다.
다른 split이나 입력을 연구하려면 별도 실험 ID·설정으로 명시하고 동일 조건 비교라고 부르지 않는다.

- Detector fit: 같은 instruction/scene의 target j 전체 제외, 나머지 4j×10판.
  40판 전체에 성공·실패가 필요하며 각 j 개별 양 클래스 조건은 추가하지 않는다.
- Operator fit: 같은 50판 중 target 성공 제외, target 실패 포함 가능. Detector와 fit pool을 혼동하지 않는다.
- Target j는 성공과 실패가 각각 1개 이상. 비슷한 조건이면 성공판이 많은 후보 우선.
  성공≥3 같은 새 최소치를 임의로 추가하지 않는다.
- 평가: scene별 target j 하나의 10판 전체를 평가한다. instruction별 목표는 3scene, arm당 30판.
  3scene 미충족은 실제 분모와 부족분을 보고하며 다른 j 반복으로 채우지 않는다.
- 기존 조건은 k8 cluster 기준, GT phase 생략, beta 0.8, DiT L12의 token-mean common shift.
  적용 layer는 학습 layer와 일치해야 하며 denoise 적용 범위는 별도 설정이다.
  재현 실행 전 실제 artifact/config와 대조하고 구 per-token 수축 결과를 같은 버전으로 재사용하지 않는다.
- 해당 phase/cluster 연산자가 없으면 reseed fallback. reseed 후 연산자 적용 arm은 이미 reseed한 pass를 유지하며 추가 reseed하지 않는다.
  공통 연산자로 fallback하는 arm은 별도 arm으로 표시한다. 실험 도중 fallback 방식을 바꾸지 않는다.
- phase/cluster별 길이 조정 여부와 episode별 기여량을 코드·artifact로 확인한다.
  실패 rollout의 후반 phase 부재는 데이터 손상이나 필수 탈락 사유가 아니다.
- 추가 baseline 재수집은 detector 개발의 자동 선행 작업이 아니다. 담당자가 명시한 평가 범위만 실행한다.

## 4. 아직 결정하지 않은 사항

| 미결 또는 누락 | 처리 |
|---|---|
| 공유 저장소 URI/권한, frozen manifest 없음 | 데이터 의존 학습·평가는 보류. 로더·인터페이스·합성 입력 검증은 가능. |
| 전달 데이터별 activation 범위 | DiT 최대 16layer×4denoise×49token×1536, 확인 저장본 7layer 범위는 detector_contract.md 참조. 전달 묶음 전체의 스키마와 VL token 추가 제공은 별도 확인. |
| detector 개발 branch/commit | 후보를 명시하고 호환성 확인. 임의로 main 최신 commit에 맞추지 않는다. |
| 개입 후 상태 정렬 | LSTM의 현행 방식을 전체 모델의 계약으로 확정하지 않는다. offline detector 개발은 진행 가능. 새로운 online 조합은 결정 전 통합 완료로 표시하지 않는다. |
| 지연 상한 | 지연을 측정해 보고. 임의 숫자로 합격 판정을 만들지 않는다. |
| 사건 라벨 없음 | 해당 snapshot의 검토 상태 확인. 기존 성공판 미라벨=무사건 합의를 새 미검토 데이터에 확장하지 않는다. |

## 5. 자원과 실행

GR00T에만 적용: A100(srv48/srv50)은 모든 세션 합산 서버별 GPU 최대 1장, GPU당 최대 6모델.
kanu는 모든 세션 합산 GPU 최대 3장, GPU당 최대 2모델. 이 수치를 다른 모델 학습의 자원 승인으로 사용하지 않는다.
발사 전 temporal_vla의 `docs/05_gpu_server_rules.md`, `configs/harness/gpu_policy.json`,
`docs/collab_codex/runbook_groot_harness.md` 최신 내용을 확인하고 공통 하네스를 사용한다.
자기 세션 전용 원장을 만들어 전체 한도를 우회하지 않는다. 다른 담당의 실행 중 job/파일을 임의 변경하지 않는다.
독립 detector 레포에 이 실행 규칙·하네스가 없으면 GR00T 실행을 통합 담당에게 넘긴다.

## 6. GitHub 연결 방식

Detector 코드 개발은 task_classification에서 진행하고, temporal_vla는 검증한 detector commit을
submodule로 고정하는 방식을 사용한다. 모델 내부 변경과 파이프라인 통합 변경을 구분한다.
전처리·모델·threshold·입력 정의가 같은 버전을 가리켜야 한다.
구체적인 후속 PR 양식, 작업 관리, AI 간 인계 방식은 이번 최초 전달 자료에서 정하지 않는다.
