"""설정 → 모델 조립. AE와 SAE는 '주입한 컴포넌트'만 다른 같은 BaseAE다.

    ae   encoder: mlp   + decoder: mlp          → 조밀 병목 AutoEncoder
    sae  encoder: topk  + decoder: linear_dict  → top-k Sparse AutoEncoder

모델 종류가 클래스가 아니라 **설정**으로 표현되므로 conf/model만 바꾸면 전환된다.

레지스트리의 각 빌더는 `(input_dim, latent_dim, **컴포넌트 고유 파라미터)`를 받는다.
공유 차원을 팩토리가 양쪽에 꽂기 때문에 인코더 출력차원과 디코더 입력차원이 어긋날 수 없다.
디코더의 출력차원은 항상 input_dim이다 (타깃이 x 자기재구성이므로).

    model = build_model(cfg.model)               # 설정 그대로
    model = build_model(cfg.model, input_dim=D)  # 데이터에서 온 차원으로 덮어쓰기
"""
from __future__ import annotations

from collections.abc import Mapping

from phase.models.autoencoder import (BaseAE, Decoder, DecoderLinearDict,
                                      DecoderSeedConditioned, Encoder, EncoderTopK,
                                      EncoderVariational, VariationalAE)
from phase.models.classifier import PhaseClassifier

ENCODERS = {
    "mlp": lambda input_dim, latent_dim, hidden=256:
        Encoder(input_dim, latent_dim, hidden),
    "variational": lambda input_dim, latent_dim, hidden=256:
        EncoderVariational(input_dim, latent_dim, hidden),
    "topk": lambda input_dim, latent_dim, k, hidden=None:
        EncoderTopK(input_dim, latent_dim, k, hidden),
}

DECODERS = {
    "mlp": lambda input_dim, latent_dim, hidden=256:
        Decoder(latent_dim, input_dim, hidden),
    "linear_dict": lambda input_dim, latent_dim:
        DecoderLinearDict(latent_dim, input_dim),
    "seed_conditioned": lambda input_dim, latent_dim, hidden=256, ctx_dim=8:
        DecoderSeedConditioned(latent_dim, input_dim, hidden, ctx_dim),
}

# 사실상 'ae' 하나다 — vae는 deprecated라 conf/model에 항목이 없다.
MODELS = {"ae": BaseAE, "vae": VariationalAE}

REQUIRED = ("input_dim", "latent_dim", "encoder", "decoder")


def _plain(obj):
    """Mapping(DictConfig 포함) → 평범한 dict. omegaconf에 의존하지 않는다."""
    return {str(k): (_plain(v) if isinstance(v, Mapping) else v)
            for k, v in obj.items()}


def _build(registry, spec, field, input_dim, latent_dim):
    """spec {'type': 이름, ...파라미터} → 모듈. type은 소비하고 나머지는 빌더로 넘긴다."""
    if not isinstance(spec, Mapping):
        raise ValueError(f"{field}는 type을 가진 매핑이어야 합니다: {spec!r}")
    spec = _plain(spec)
    name = spec.pop("type", None)
    if name not in registry:
        raise ValueError(f"{field}.type은 {tuple(registry)} 중 하나여야 합니다: {name!r}")
    try:
        return registry[name](input_dim=input_dim, latent_dim=latent_dim, **spec)
    except TypeError as e:
        # 오타·누락 파라미터는 빌더 호출에서 잡힌다. 어느 컴포넌트인지만 얹어준다
        # (빌더가 람다라 원본 메시지만으로는 위치를 알 수 없다).
        raise TypeError(f"{field}(type={name!r}) 조립 실패: {e}") from e


def build_model(cfg, input_dim=None):
    """model 설정 블록 → 모델.

    [인자]
        cfg        model 블록 (dict 또는 DictConfig). REQUIRED 키가 있어야 한다.
        input_dim  주면 cfg의 input_dim을 덮어쓴다 — 실제 데이터 차원이 설정값과
                   다를 때 설정 파일을 고치지 않고 맞추기 위한 통로.
    """
    if not isinstance(cfg, Mapping):
        raise ValueError(f"model 설정은 매핑이어야 합니다: {cfg!r}")
    missing = [k for k in REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"model 설정에 {missing} 키가 없습니다")

    D = cfg["input_dim"] if input_dim is None else input_dim
    d = cfg["latent_dim"]

    enc = _build(ENCODERS, cfg["encoder"], "encoder", D, d)
    dec = _build(DECODERS, cfg["decoder"], "decoder", D, d)

    kind = cfg.get("kind", "ae")
    if kind not in MODELS:
        raise ValueError(f"model.kind는 {tuple(MODELS)} 중 하나여야 합니다: {kind!r}")
    # loss/beta는 모델 클래스가 기본값과 검증을 갖고 있으므로 있을 때만 넘긴다
    kwargs = {k: cfg[k] for k in ("loss", "beta") if k in cfg}
    return MODELS[kind](enc, dec, **kwargs)


def build_classifier(cfg, input_dim=None, n_class=None):
    """지도 분류기 조립. AE/SAE와 달리 enc/dec 쌍이 아니라 trunk+head 한 덩어리라
    build_model과 분리한다 (구조가 다른 것을 억지로 같은 팩토리에 끼우지 않는다).

        model = build_classifier(cfg.model, input_dim=D, n_class=13)
    """
    if not isinstance(cfg, Mapping):
        raise ValueError(f"model 설정은 매핑이어야 합니다: {cfg!r}")
    D = cfg["input_dim"] if input_dim is None else input_dim
    n = cfg["n_class"] if n_class is None else n_class
    return PhaseClassifier(D, n, hidden=cfg.get("hidden", 256),
                           depth=cfg.get("depth", 2), dropout=cfg.get("dropout", 0.0))
