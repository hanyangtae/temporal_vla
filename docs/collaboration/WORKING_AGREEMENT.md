# Detector 개발 시 주의사항

- 대상: 김상우 및 detector 개발 AI
- 용도: 협업 설명용 요약. 구현 세부 조건은 연결 문서 참고

## 1. 목표와 역할

- **목표: 오발화 감소 · 실패 감지 유지 · 사건 전 발화 개선**
- 사건 후 stuck 감지 능력은 별도 평가
- **김상우:** `task_classification`에서 detector 개발
- **박경태:** steering 및 전체 파이프라인 통합
- 모델 구조 자유: LSTM·두 MLP 등 특정 구조로 제한하지 않음

## 2. 입력과 출력

| 항목 | 합의한 내용 |
|---|---|
| 입력 | Activation. 제공 범위 안에서 layer·denoise step·token 선택과 pooling 가능 |
| 시간 단위 | Inference step. |
| 출력 | **현재 inference step의 개입 필요 여부** |
| 판정 시점 | **해당 action이 환경에서 실행되기 전**, 추후 activation을 수정하고 denoise step을 다시 밟을 만큼에 여유도 필요  |

- 저장본: **`[7 layers, 4 denoise steps, 49 tokens, 1536 features]`**
- 저장 layer: `[0, 2, 4, 8, 10, 12, 15]`
- 코드상 캡처 범위: 전체 16개 layer. 미저장 layer는 추가 캡처 필요
- 전달 묶음별 실제 저장 범위 확인

→ [축별 범위·token 구성·온라인 연결 조건](detector_contract.md)

## 3. 같은 데이터인지 확인할 재현 조건

- **데이터는 승준 서버에 접속해 가져오기**
- [기본/추가 수집·라벨 정본·라벨러 경로](data_contract.md#데이터-위치--승준-서버) 참고
- 접속 정보·권한은 박경태에게 확인

- **Task·instruction:** task/env 이름, instruction 원문, 좌우 키 매핑
- **Scene:** plan, scene_idx, 실제 layout_id·style_id, 주방 후보 목록과 순서
- **객체·fixture:** 종류, asset 버전, target 매핑, 초기 pose·관절 상태
- **Jitter:** jitter_idx(j), reset_idx, lat/back 오프셋, base pose, 좌표계·부호
- **Noise·seed:** noise_idx(n), env/scenario seed, inference/policy seed, seed 매핑·reseed 설정
- **초기 상태:** ep_meta, simulator state, robot qpos/qvel·gripper, reset 순서·재시도·warmup
- **Episode 실행:** episode_idx·시작 index, episode 수, 실행 순서, 환경 재사용 여부, one_episode_per_env
- **병렬 환경:** n_envs, worker 수, episode→worker 배정, 동기/비동기 실행, policy batch 크기
- **Machine:** 실제 수집·policy serve·render 머신 및 각 실행 device
- **하드웨어·런타임:** GPU 모델, driver, CUDA/cuDNN, container·라이브러리 버전, thread 수
- **실행 코드:** collector·serve·policy·benchmark·submodule 버전, 미커밋 변경, 실제 로드 경로
- **정책 모델:** checkpoint, model config, embodiment, eval mode, dtype·TF32·결정성 설정
- **관측 입력:** camera 이름·순서·pose·해상도, crop/resize/회전·색상 처리, state 항목·정규화, 관측 history
- **정책 추론:** action horizon, denoise 횟수·schedule, sampling 설정, 실제 noise, RNG 초기화·소비 순서
- **Action 실행:** n_action_steps, queue/reset, 정규화·역정규화, rotation/gripper·좌표계·clipping
- **물리 진행:** controller, control rate, physics timestep·substep, horizon, 성공·종료·timeout 조건
- **개입 조건:** arm, detector·전처리·threshold, operator·beta·적용 위치, gate·fallback·reseed·재-forward·상태 갱신
- **렌더링:** backend(EGL/OSMesa 등), vendor(NVIDIA/Mesa 등), render device, camera·해상도·렌더 옵션
- **영상 기록:** FPS, steps_per_render, 시작 offset, frame 누락·중복, 전체 frame 수·PTS, codec·pixel format·편집 설정
- **Activation 저장:** hook 위치, capture layer·denoise slot·token, pooling·dtype, 후보/개입 pass, inference record와 영상 frame 대응

같은 조건의 결정적 실행은 같은 결과를 기대하며, 실제 궤적·영상 일치는 별도 검증.
식별값 용도와 상세 확인 방법은 [데이터 문서](data_contract.md#식별값별-용도) 참고.

## 4. 학습 주의사항

### ① 데이터 누출 방지

- 같은 rollout의 frame·복사본을 train/test로 나누지 않기
- 표준화·PCA·AE·cluster의 fit 데이터 확인
- Threshold 선택·calibration과 최종 평가 데이터 구분
- 기존 target-j 비교: **대상 j 10판 전체 제외 → 다른 4j 40판으로 학습**
- 대상 j의 실패판도 학습에서 제외
- 40판 전체에 양 클래스 필요. 각 j별 양 클래스는 필수 아님
- 새 split 사용 가능. 기존과 다른 실험임을 명시

### ② 길이 편향·미래 정보 방지

- 긴 episode의 과도한 학습 기여 확인
- 성공/실패의 길이 차이만 학습하는지 확인
- 미래 activation·전체 episode 평균·종료 후 확정되는 길이를 온라인 입력에 사용 금지

### ③ 최종 결과와 사건 라벨 구분

- 최종 실패 rollout의 모든 frame을 실패 상태로 해석하지 않기
- 성공 rollout에도 실패 사건·회복 가능
- 정상 timeout·후반 phase 미도달을 데이터 손상으로 제외 금지

→ [데이터 분할·길이 통제·평가 상세](data_contract.md#detector-학습평가-상세-주의사항)

## 5. 평가 항목

| 보고할 것 | 구분할 점 |
|---|---|
| 성공판 발화 | 성공판 전체와 **검토된 무사건 성공판**을 구분 |
| 실패판 감지 | 발화율과 미발화 수 |
| 발화 시점 | 첫 사건 전/후와 시차. 사후 stuck 감지는 사전 예측과 구분 |
| 온라인 동작 | Action 실행 전 판정, 지연, episode 간 상태 reset |

- 모든 지표에 **분자·분모·미검토/누락 수** 표시
- Frame 지표와 rollout 지표 구분
- Instruction/scene별 편차 확인
- Detector 발화 성능과 steering 구제 성능 구분

## 6. 미결 사항과 연결 방식

- 실제 사용할 데이터·라벨·split snapshot과 승준 서버 접속 권한 확인.
- 사용할 detector 개발 branch/commit과 허용 지연.
- **개입 후 detector 상태 처리:** 현재 LSTM의 hidden-state 갱신을 다른 모델에도 그대로 적용할지는 미결.

- 코드 개발: `task_classification`
- 통합: 검증한 commit을 `temporal_vla` submodule로 연결
- 모델·전처리·threshold·입력 정의를 함께 고정
- 추가 캡처·파이프라인 변경은 박경태와 조율
- **데이터가 추가로 필요하면 박경태에게 요청**: 필요한 task/scene, 성공·실패 구성, 수량, activation 범위와 사용 목적을 함께 전달
---

- **jitter(j):** 같은 scene의 위치 변화
- **noise(n):** 같은 scene/jitter의 정책 diffusion noise 변화
