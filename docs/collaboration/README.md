# 김상우의 Detector 개발을 위한 최초 전달 자료

목적: 김상우와 함께 작업하는 AI가 이전 대화 없이 detector 개발 시 지킬 입력·데이터·학습·평가 조건을 이해하도록 한다.
후속 작업 관리나 AI 간 세션 인계 절차는 이 자료의 범위가 아니다.

## 읽는 순서

1. [공동 작업 조건](WORKING_AGREEMENT.md): detector 개발 목표, 학습·평가 주의사항, 사용자 확정 조건과 미결 사항.
2. [데이터·라벨](data_contract.md): rollout 식별, 공유 방식, 라벨·split 버전 관리.
3. [Detector 인터페이스](detector_contract.md): 입력 범위, 현재 inference step의 발화 판정, 개입 후 상태 정렬의 고민점.

## 전달할 핵심

- 전체 파이프라인과 steering은 temporal_vla에서, detector 개발은 task_classification에서 담당한다.
- task_classification은 이미 submodule이다. 확인 시 고정 commit은
  `88543a2a00e37a1de409ccdaaaa865ab787c7196`이었다. 새 detector 개발 branch/commit은 아직 결정하지 않았다.
- Detector 입력은 activation을 기반으로 유연하게 선택한다. 실제 제공 가능한 capture 범위를 확인해 공유한다.
- 출력은 현재 inference step의 fire/no-fire이며 해당 action이 환경에서 실행되기 전에 판정해야 한다.
- 개입 후 detector 상태를 어떻게 정렬할지는 미결이다. 기존 LSTM 방식을 새 모델 전체에 강제하지 않는다.

## 최초 전달 전에 채울 정보

- [승준 서버 데이터 경로](data_contract.md#데이터-위치--승준-서버) 확인 후 접속해 취득. 접속 정보·권한은 박경태에게 확인.
- 전달할 데이터 manifest, 라벨 snapshot, split 및 작은 입력 예시.
- 전달 대상별 activation 스키마 점검과 실행 환경. DiT의 확인된 축 범위는 [입력 표](detector_contract.md)에 명시했다.
- 동료가 작업할 detector branch/commit.

서버 경로는 제공됐으며 사용할 manifest·snapshot·split은 아직 확보·확정되지 않았다. 이 문서만으로 데이터 전달이나 online 연결까지 완료된 것은 아니다.
이번 준비에서는 동료 레포 수정, GitHub 게시, AI 진입점 설치를 하지 않았다.
