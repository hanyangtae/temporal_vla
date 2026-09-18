# Detector 개발 시 주의사항

김상우와 detector 개발 방향을 맞추기 위한 요약입니다. 구현 시 세부 조건은 연결된 문서를 참고합니다.

## 1. 무엇을 개발하나

**오발화를 줄이면서 실패 감지를 유지하고, 실패 사건 전에 더 일찍 발화하는 detector**를 개발합니다.
이미 실패에 stuck된 상태를 감지하는 능력도 따로 봅니다.

- **김상우:** `task_classification`에서 detector 개발.
- **박경태:** steering과 전체 파이프라인 통합.
- 모델 구조는 자유입니다. LSTM이나 두 MLP로 제한하지 않습니다.

## 2. 입력과 출력

| 항목 | 합의한 내용 |
|---|---|
| 입력 | Activation. 제공 범위 안에서 layer·denoise step·token 선택과 pooling 가능 |
| 시간 단위 | Inference step. Denoise step과 구분 |
| 출력 | **현재 inference step의 개입 필요 여부** |
| 판정 시점 | **해당 action이 환경에서 실행되기 전** |

확인된 저장본은 **`[7 layers, 4 denoise steps, 49 tokens, 1536 features]`**입니다.
저장 layer는 `[0, 2, 4, 8, 10, 12, 15]`이고, 코드상 전체 16개 layer를 캡처할 수 있습니다.
기존 파일에 없는 layer는 추가 캡처가 필요합니다. 모든 기존 데이터가 같은 범위인지는 전달 묶음별로 확인합니다.

→ [축별 범위·token 구성·온라인 연결 조건](detector_contract.md)

## 3. 같은 데이터인지 먼저 확인

**같은 scene/j/n이나 seed만으로 activation·영상·라벨을 연결하면 안 됩니다.**
원본 실행과 파일의 관계를 확인해야 합니다.

| 식별값 | 무엇을 확인하나 |
|---|---|
| `pkl_sha256` | 같은 PKL 파일인지, 손상·교체되지 않았는지 |
| `sig` | PKL을 찾는 짧은 값. 현재는 SHA256 앞 16자리이며 전체 hash와 함께 확인 |
| `video_sha256` | 같은 영상 파일인지 |
| `run_id` | 어느 실행에서 생성됐는지 |

PKL hash로 연결된 영상을 **찾을 수는 있지만**, 그 값 자체가 영상 hash는 아닙니다.
또한 PKL은 metadata 차이만으로도 hash가 달라질 수 있어 **파일 차이와 궤적 차이를 구분**해야 합니다.
`video_sha256`과 `run_id`가 모든 기존 파일에 이미 저장돼 있다는 뜻은 아닙니다.

동일 조건의 결정적 재실행은 동일 결과를 기대합니다. Run ID는 따로 두되 실제 궤적·영상 일치를 확인해
원본과 연결합니다. 같은 내용으로 확인된 재현본은 독립 표본 두 개로 세지 않습니다.

<details>
<summary>재실행 시 대조할 조건 펼치기</summary>

- **원본:** plan·run ID, 실제 생성 머신, PKL·영상·activation hash와 파생 이력.
- **환경:** task·instruction·좌우 키 의미, 실제 layout/style, 객체·fixture, jitter의 reset/위치/좌표계.
- **초기 상태·난수:** ep_meta, simulator·robot·객체 상태, reset 순서, env/policy seed, RNG 소비 순서·실제 noise.
- **정책·실행 환경:** checkpoint·전처리·코드·라이브러리, GPU·driver·renderer·수치 정밀도와 결정성 설정.
- **관측·action:** camera·영상 전처리·state/history, horizon·denoise·action 실행 수·후처리·물리 timestep·종료/개입 조건.
- **영상·activation:** frame 시각·누락·FPS·codec·편집 이력, hook·pooling·dtype·capture pass, frame↔env step↔inference step 대응.

설정 일치는 실제 영상 동일성의 충분조건이 아닙니다. 영상 파일 hash 또는 정확한 프레임·시각 비교로 확인합니다.
재인코딩본은 파생 영상이고, 원본의 byte-copy는 저장 머신이 달라도 같은 파일입니다.
빠진 조건은 추정하지 않고 미확인으로 표시합니다.

</details>

→ [전체 대조표와 영상·라벨 연결 기준](data_contract.md#재실행-조건-대조표)

## 4. 학습할 때 중요한 세 가지

**① 평가 데이터가 학습에 들어가지 않게 합니다.**
같은 rollout의 frame이나 복사본을 train/test에 나누지 않습니다. 표준화·PCA·AE·cluster 등 전처리의 fit 데이터도 확인합니다.
Threshold 선택·calibration과 최종 평가도 구분합니다.

기존 target-j 비교 기준은 **대상 j의 10판 전체를 제외하고 다른 4j의 40판으로 학습**하는 것입니다.
대상 j의 실패판도 제외합니다. 40판 전체에 양 클래스가 필요하지만 각 j마다 양 클래스일 필요는 없습니다.
새 split은 사용할 수 있으나 기존과 다른 실험임을 명시합니다.

**② 길이 차이와 미래 정보에 주의합니다.**
긴 episode가 학습을 지배하거나 성공/실패의 길이 차이만 배우지 않는지 봅니다.
현재 시점 이후 activation, 전체 episode 평균·최종 길이 등은 온라인 입력으로 사용할 수 없습니다.

**③ 최종 실패와 ‘현재 실패 사건’을 구분합니다.**
실패한 rollout의 모든 frame이 실패 상태는 아닙니다. 성공한 rollout에도 실패 사건과 회복이 있을 수 있습니다.
정상 timeout이나 후반 phase 미도달을 데이터 손상으로 제외하지 않습니다.

→ [데이터 분할·길이 통제·평가 상세](data_contract.md#detector-학습평가-상세-주의사항)

## 5. 성능은 이렇게 나눠 봅니다

| 보고할 것 | 구분할 점 |
|---|---|
| 성공판 발화 | 성공판 전체와 **검토된 무사건 성공판**을 구분 |
| 실패판 감지 | 발화율과 미발화 수 |
| 발화 시점 | 첫 사건 전/후와 시차. 사후 stuck 감지는 사전 예측과 구분 |
| 온라인 동작 | Action 실행 전 판정, 지연, episode 간 상태 reset |

모든 수치는 **분자·분모와 미검토/누락 수**를 함께 보고합니다. Frame 지표와 rollout 지표,
instruction/scene별 편차도 구분합니다. Detector 발화 성능과 steering의 구제 성능은 별개입니다.

## 6. 아직 정해야 할 것

- 실제 전달할 데이터·라벨·split snapshot과 접근 방법.
- 사용할 detector 개발 branch/commit과 허용 지연.
- **개입 후 detector 상태 처리:** 현재 LSTM의 hidden-state 갱신을 다른 모델에도 그대로 적용할지는 미결.

코드는 `task_classification`에서 개발하고, 검증한 commit을 `temporal_vla`의 submodule로 연결합니다.
모델·전처리·threshold·입력 정의를 함께 고정합니다. 추가 캡처나 파이프라인 변경은 박경태와 맞춥니다.

---

**용어:** jitter(j)는 같은 scene의 위치 변화, noise(n)는 같은 scene/jitter의 정책 diffusion noise 변화입니다.
