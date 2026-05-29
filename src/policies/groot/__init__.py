"""GR00T 모델 wrapper helpers.

서빙 (`scripts/serve/groot.py`) 과 학습 어댑터 양쪽에서 공유하는 utility.
GR00T 모델 자체는 submodule `src/policies/Isaac-GR00T` 에 있고, 이 패키지는
그 위에 얹는 데이터/모델 변환 layer.

- preprocess: 이미지 변환 (square pad + 256 resize) 등
- schema:     통일 API ↔ GR00T schema 키 매핑 + modality key 정규화
- robocasa_io: GrootRoboCasaEnv native obs/action ↔ HTTP GR00T adapter
- loader:     Gr00tSimPolicyWrapper 로딩 + statistics.json 파싱
- service:    HTTP serving runtime state + act/feature/health behavior
"""
