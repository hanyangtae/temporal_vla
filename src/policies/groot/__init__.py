"""GR00T N1.6 wrapper helpers.

GR00T 모델 자체는 submodule `src/policies/Isaac-GR00T` 에 있고, 이 패키지는
N1.6 위에 얹는 데이터/모델 변환 layer다. GR00T N1.5 LeRobot serving 경로는
`scripts/serve/lerobot_adapters/groot.py`에 둔다.

- core:      로더, schema, 이미지 전처리, HTTP service runtime
- robocasa:  GrootRoboCasaEnv native obs/action, replay, eval wrappers
- safe:      N1.6 DiT SAFE feature capture/serialization helpers

Cross-policy SAFE feature metadata naming lives in `src/policies/safe_metadata.py`.
"""
