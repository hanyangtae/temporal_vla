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
| 시간 단위 | Inference step. Denoise step과 구분 |
| 출력 | **현재 inference step의 개입 필요 여부** |
| 판정 시점 | **해당 action이 환경에서 실행되기 전** |

- 확인 저장본: **`[7 layers, 4 denoise steps, 49 tokens, 1536 features]`**
- 저장 layer: `[0, 2, 4, 8, 10, 12, 15]`
- 코드상 캡처 범위: 전체 16개 layer. 미저장 layer는 추가 캡처 필요
- 전달 묶음별 실제 저장 범위 확인

→ [축별 범위·token 구성·온라인 연결 조건](detector_contract.md)

## 3. 같은 데이터인지 먼저 확인

- **scene/j/n·seed만으로 activation·영상·라벨 연결 금지**
- 원본 실행과 각 파일의 대응 관계 확인

| 식별값 | 무엇을 확인하나 |
|---|---|
| `pkl_sha256` | 같은 PKL 파일인지, 손상·교체되지 않았는지 |
| `sig` | PKL을 찾는 짧은 값. 현재는 SHA256 앞 16자리이며 전체 hash와 함께 확인 |
| `video_sha256` | 같은 영상 파일인지 |
| `run_id` | 어느 실행에서 생성됐는지 |

- PKL hash: 대응 영상 조회에 사용 가능. 영상 동일성 자체의 증거는 아님
- Metadata 차이만으로 PKL hash가 바뀔 수 있음 → **파일 차이 ≠ 궤적 차이**
- 기존 파일의 `video_sha256`·`run_id` 보유 여부 확인
- 동일 조건의 결정적 재실행: 동일 결과 기대, run ID는 별도 부여
- 원본 연결 전 실제 궤적·영상 일치 검증
- 동일 내용의 재현본을 독립 표본으로 중복 집계 금지

<details>
<summary>재실행 시 대조할 조건 펼치기</summary>

- **원본:** plan·run ID, 실제 생성 머신, PKL·영상·activation hash와 파생 이력.
- **환경:** task·instruction·좌우 키 의미, 실제 layout/style, 객체·fixture, jitter의 reset/위치/좌표계.
- **초기 상태·난수:** ep_meta, simulator·robot·객체 상태, reset 순서, env/policy seed, RNG 소비 순서·실제 noise.
- **정책·실행 환경:** checkpoint·전처리·코드·라이브러리, GPU·driver·renderer·수치 정밀도와 결정성 설정.
- **관측·action:** camera·영상 전처리·state/history, horizon·denoise·action 실행 수·후처리·물리 timestep·종료/개입 조건.
- **영상·activation:** frame 시각·누락·FPS·codec·편집 이력, hook·pooling·dtype·capture pass, frame↔env step↔inference step 대응.

- 설정 일치만으로 동일 영상 판정 금지: 영상 hash 또는 프레임·시각 비교 필요
- 재인코딩본: 파생 영상으로 구분
- 원본 byte-copy: 저장 머신이 달라도 동일 파일
- 누락된 조건: 추정하지 않고 미확인 표시

</details>

→ [전체 대조표와 영상·라벨 연결 기준](data_contract.md#재실행-조건-대조표)

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

- 실제 전달할 데이터·라벨·split snapshot과 접근 방법.
- 사용할 detector 개발 branch/commit과 허용 지연.
- **개입 후 detector 상태 처리:** 현재 LSTM의 hidden-state 갱신을 다른 모델에도 그대로 적용할지는 미결.

- 코드 개발: `task_classification`
- 통합: 검증한 commit을 `temporal_vla` submodule로 연결
- 모델·전처리·threshold·입력 정의를 함께 고정
- 추가 캡처·파이프라인 변경은 박경태와 조율

---

- **jitter(j):** 같은 scene의 위치 변화
- **noise(n):** 같은 scene/jitter의 정책 diffusion noise 변화
