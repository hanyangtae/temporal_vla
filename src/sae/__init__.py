"""top-k SAE 코어 — scene-feature 분리(exp5 / G1) 라이브러리.

출처: 동료 레포 `task_classification@88543a2` (https://github.com/robots-oh/task_classification)
      의 `phase/{models,train,clustering,metrics,data}` 코어를 lift 한 것.
      hydra·wandb·omegaconf 의존을 제거하고 torch/numpy/sklearn 만 남겼다.
검토 보고서: docs/steering/29_sae_port_review.md (파일:라인 근거)
이식 계획:   docs/steering/30_sae_g1_port_handout.md §2.1, §4 Phase A

[구성]
    models.py   EncoderTopK / DecoderLinearDict / BaseAE + build_model 팩토리
    train.py    미니배치 학습 루프 + val-loss early stopping (seed·device 인자)
    cluster.py  사후 이산화(KMeans/GMM) + dead-feature 처리
    metrics.py  U-coefficient · purity · silhouette + clock(시간분위) 기준선
    pca.py      train-only PCA-whiten — **G1 에선 미사용**(원본 1536-d 직접 입력)

[의도적으로 가져오지 않은 것]
    phase/data/make_phase_dataset.py  — 토큰 축을 평균으로 없앤다(`:102-109`).
        우리는 per-token 보존이 계약이므로 재사용 금지, 빌더는 신규 작성
        (scripts/scene_sae/build_sae_inputs.py, 핸드아웃 §2.2-1 / §4 Phase B).
    phase/models/classifier.py, metrics/{boundary,self_transition}.py — phase 지도 트랙 전용.
    phase/clustering/gpu.py — sklearn 경로만 쓴다.

[의존성 제약] scipy 를 쓰지 말 것 — 분석·fit 을 돌리는 원격 노드(승준)에 scipy 가 없다.
             (tests/test_sae_core.py 가 이 규칙을 검사한다.)

참고: 레포의 `scripts/event_sae/` 는 영상(MP4) 프레임용 Event-SAE 어댑터로 **무관한 별개 라인**이다.
"""
from src.sae.cluster import active_mask, assign, dead_fraction, fit_clusters
from src.sae.metrics import (DEFAULT_METRICS, EvalContext, Metric, Purity,
                             Silhouette, UncertaintyCoefficient, clock_clusters,
                             derived_flags, episode_bounds, evaluate, purity,
                             silhouette, time_fraction, uncertainty_coef)
from src.sae.models import (BaseAE, Decoder, DecoderLinearDict, Encoder,
                            EncoderTopK, build_model, sae_config)
from src.sae.train import (encode_all, epoch, fit, make_optimizer, set_seed,
                           train_sae)

__all__ = [
    # models
    "BaseAE", "Decoder", "DecoderLinearDict", "Encoder", "EncoderTopK",
    "build_model", "sae_config",
    # train
    "set_seed", "make_optimizer", "epoch", "fit", "train_sae", "encode_all",
    # cluster
    "fit_clusters", "assign", "active_mask", "dead_fraction",
    # metrics
    "EvalContext", "Metric", "UncertaintyCoefficient", "Purity", "Silhouette",
    "uncertainty_coef", "purity", "silhouette", "clock_clusters",
    "episode_bounds", "time_fraction", "DEFAULT_METRICS", "derived_flags", "evaluate",
]
