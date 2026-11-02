"""Fail-closed loader 계약: (A) GR00T finetuning 데이터 → QAM/RL2 학습, (B) 우리 rollout → TRQAM.

이 모듈은 split 을 만들지 않는다. 데이터 정체·episode 구성·split·시나리오는 메인 세션이 manifest 에
``status: confirmed`` 로 확정해 넘겨야 하며, 미확정이면 ``DataContractBlocked`` 로 멈춘다.
feature 추출은 docs/04 의 ``feature_axes`` 로 판정한다 (ndim 금지).

QAM 관측은 action_decoder 입력 last16 평균 1024 (composition.SPEC). 기존 rollout pkl 의
``[7, 4, 49, 1536]`` block residual capture 는 다른 인터페이스라 이 계약에서 거부한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from composition import FEATURE_CAPTURE_POINT, FEATURE_DIM, SPEC
from contract import (
    ACTION_HORIZON,
    CONFIG_SHA256,
    ContractError,
    DIT_SEQ_LEN,
    DIT_WIDTH,
    METADATA_SHA256,
    VLM_CONTEXT_DIM,
    sha256_file,
)

CONTRACT_VERSION = "groot_robocasa_adapter_20261021/v2"
ALLOWED_TRAIN_SPLITS = ("train", "val")

# checkpoint-120000 학습 데이터 계보 (outputs/analysis/checkpoint_training_data_20260916 +
# outputs/analysis/groot_robocasa_lineage_20261021). manifest 가 이 값과 다르면 거부한다.
FINETUNE_LINEAGE = {
    "checkpoint": "robocasa/robocasa365_checkpoints:gr00t_n1-5/multitask_learning/checkpoint-120000",
    "dataset_soup": "pretrain_human300",       # Tabletop24k 아님
    "tasks": 300,                              # atomic 65 + composite 235
    "nominal_demos": 30000,                    # filter_key 100_demos × 300 task (공식 설명값)
    "filter_key": "100_demos",
    "registry_commit": "4f8a2980def75a55dff96b990745b83540425f09",
    # trainer_state: global_step 120000, max_steps 300000(계획 상한), batch 128.
    # 저장된 120000 은 중간 checkpoint 이며 300000 상한과 모순이 아니다.
}
REJECTED_DATASET_SOUPS = ("tabletop24k", "target500", "mimicgen")
# 로컬 OpenDrawer/20250819 LeRobot 실측 (meta/info.json sha256 73367b37...). train 0:102 만 있다.
LOCAL_OPENDRAWER = {"episodes": 102, "frames": 20488, "splits": {"train": "0:102"},
                    "info_sha256": "73367b370772d91770a5ca6c987ad1ed95777111d440cacf34a99d6c99e02506"}
FINETUNE_UNRESOLVED = (
    "historical 100_demos 필터가 로컬 102 episode 중 정확히 어느 100 개였는지 (membership)",
)
LOCAL_OPENDRAWER["episodes_jsonl_sha256"] = (
    "59b4d478a31e613f99d36c26baaec54a0222d9b04a76f50a64b9c8a1d963b085")
OPENDRAWER_PACKAGE_REL = "robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot"

# 학습 데이터 정체를 어디까지 주장하는지 (load_finetune_data_manifest 의 purpose).
# - original_full_reproduction: checkpoint-120000 의 원래 100_demos episode 목록 그대로. 그 목록은
#   아직 모른다 → historical_exact membership 증거가 없으면 차단.
# - published_package_subset_for_smoke: 공개 OpenDrawer 20250819 package 102 개 전부를 inventory 로
#   검증하고 새 92/10 split 으로 쓴다. 원래 100 개 membership 은 주장하지 않는다 (smoke/보조 정책용).
# - published_package_subset_for_baseline: 같은 package·split 으로 production QAM baseline 을 학습한다.
#   증거 요건은 smoke 와 같고 mode 이름만 다르다 (smoke manifest 를 baseline 에 쓰거나 그 반대는 거부).
#   역시 원래 100 개 membership 은 주장하지 않는다.
MEMBERSHIP_PURPOSES = {
    "original_full_reproduction": "historical_exact_100_demos",
    "published_package_subset_for_smoke": "published_package_subset_for_smoke",
    "published_package_subset_for_baseline": "published_package_subset_for_baseline",
}
PACKAGE_SUBSET_PURPOSES = ("published_package_subset_for_smoke", "published_package_subset_for_baseline")
# 메인이 확정한 OpenDrawer aux split v2 (v1 과 membership·salt·parquet hash 동일, 경로 이식성만 변경).
AUX_SPLIT_V2_REL = "outputs/analysis/groot_robocasa_lineage_20261021/OpenDrawer_aux_split_v2.json"
AUX_SPLIT_V2_SHA256 = "7959f2954cd38a86c64bbef605815fc745c00c9edc37ae5f25851abb2f25af64"
AUX_SPLIT_STATUS = "prepared_new_auxiliary_split_not_original_training_split"

# TRQAM adaptation 에 넣을 수 있는 episode 역할.
ADAPT_ROLES = ("historical_base", "historical_expert", "changed_failure")
REJECTED_ROLES = ("changed_success",)


class DataContractBlocked(ContractError):
    """데이터 정체/split/feature 계약 미확정 — 학습 금지."""


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise DataContractBlocked(msg)


def _confirmed(block: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(block, Mapping), f"{name} 블록 없음")
    _require(block.get("status") == "confirmed",
             f"{name}.status={block.get('status')!r} (메인 세션 확정 필요)")
    return block


def _check_common(m: Mapping[str, Any], feature_signature: str) -> None:
    _require(m.get("contract_version") == CONTRACT_VERSION,
             f"contract_version {m.get('contract_version')!r} != {CONTRACT_VERSION}")
    ck = m.get("checkpoint") or {}
    _require(ck.get("config_sha256") == CONFIG_SHA256 and
             ck.get("metadata_sha256") == METADATA_SHA256,
             "checkpoint sha 가 checkpoint-120000 고정값과 다름")
    fe = _confirmed(m.get("feature_extraction"), "feature_extraction")
    _require(feature_signature == SPEC.signature(),
             "호출자 feature_signature 가 decoder-입력 계약 signature 와 다름")
    _require(fe.get("feature_signature") == feature_signature,
             f"feature_signature {fe.get('feature_signature')!r} != {feature_signature}")
    _require(fe.get("capture_point") == FEATURE_CAPTURE_POINT,
             f"capture_point {fe.get('capture_point')!r} != {FEATURE_CAPTURE_POINT} "
             "(block residual 등 다른 interface 거부)")
    _require(fe.get("denoise_step") == 0, "feature_extraction.denoise_step 는 0 이어야 함")


def _check_episodes(eps: Any, allowed_splits: Sequence[str]) -> list[Mapping[str, Any]]:
    _require(isinstance(eps, list) and len(eps) > 0, "episodes 비어 있음")
    ids = [e.get("episode_id") for e in eps]
    _require(all(isinstance(i, str) and i for i in ids), "episode_id 누락")
    _require(len(set(ids)) == len(ids), "episode_id 중복")
    bad = sorted({str(e.get("split")) for e in eps} - set(allowed_splits))
    _require(not bad, f"허용되지 않은 split {bad} (허용 {list(allowed_splits)}); split 임의 생성 금지")
    _require(any(e.get("split") == "train" for e in eps), "train episode 없음")
    groups = {}
    for e in eps:
        g = e.get("run_group")
        _require(isinstance(g, str) and g, f"{e['episode_id']}: run_group 없음")
        groups.setdefault(g, set()).add(e["split"])
    leak = sorted(g for g, s in groups.items() if len(s) > 1)
    _require(not leak, f"run_group 이 train/val 양쪽에 걸침: {leak}")
    return eps


def _check_membership(m: Mapping[str, Any], purpose: str) -> dict:
    """episode membership 증거 모드 판정. 어떤 모드도 상대 모드의 주장을 대신하지 못한다."""
    _require(purpose in MEMBERSHIP_PURPOSES,
             f"purpose {purpose!r} not in {sorted(MEMBERSHIP_PURPOSES)} (데이터 정체 주장 범위 명시 필요)")
    mem = _confirmed(m.get("episode_membership"), "episode_membership")
    mode = mem.get("mode")
    want = MEMBERSHIP_PURPOSES[purpose]
    if purpose == "original_full_reproduction":
        _require(mode == want,
                 f"episode_membership.mode={mode!r}: original full reproduction 은 historical 100_demos "
                 "episode 목록(현재 미상)이 필요하다. published package subset 은 원래 membership 이 아님")
        _require(bool(mem.get("evidence")) and isinstance(mem.get("historical_episode_list_sha256"), str),
                 "historical_exact membership 은 evidence + historical_episode_list_sha256 필요")
        ids = sorted(e.get("episode_id", "") for e in m.get("episodes") or [])
        got = hashlib.sha256("\n".join(ids).encode()).hexdigest()
        _require(got == mem["historical_episode_list_sha256"],
                 "manifest episode 목록이 historical_episode_list_sha256 과 다름")
        return {"purpose": purpose, "original_membership_confirmed": True,
                "data_identity_claim": "historical 100_demos exact membership (evidence 기반)"}
    _require(mode == want, f"episode_membership.mode={mode!r} != {want!r}")
    _require(mem.get("historical_membership_claimed") is False,
             "published package subset 모드는 historical_membership_claimed=false 를 명시해야 함 "
             "(원래 100 개 membership 주장 금지)")
    inv = mem.get("package_inventory") or {}
    for k in ("info_sha256", "episodes_jsonl_sha256"):
        _require(inv.get(k) == LOCAL_OPENDRAWER[k], f"package_inventory.{k} 가 로컬 package 실측과 다름")
    _require(inv.get("package_episode_count") == LOCAL_OPENDRAWER["episodes"],
             "package_inventory.package_episode_count != 102")
    _require(isinstance(mem.get("aux_split_sha256"), str), "aux_split_sha256 (split 파일 지문) 필요")
    use = "smoke 전용" if purpose == "published_package_subset_for_smoke" else "보조 정책 baseline 용"
    return {"purpose": purpose, "original_membership_confirmed": False,
            "data_identity_claim": "published OpenDrawer 20250819 package (102) 의 새 92/10 split — "
                                   f"원래 100_demos membership 아님, {use}"}


def load_finetune_data_manifest(path: str | Path, *, feature_signature: str, purpose: str) -> dict:
    """(A) checkpoint-120000 finetuning 데이터 manifest.

    purpose 로 데이터 정체 주장 범위를 정한다 (MEMBERSHIP_PURPOSES). 반환 dict 의
    ``membership_claim`` 에 실제로 주장 가능한 범위를 기록한다.
    """
    m = json.loads(Path(path).read_text())
    _check_common(m, feature_signature)
    ident = _confirmed(m.get("dataset_identity"), "dataset_identity")
    soup = str(ident.get("dataset_soup", "")).lower()
    _require(soup not in REJECTED_DATASET_SOUPS,
             f"dataset_soup {soup!r} 은 checkpoint-120000 학습 데이터가 아님 (pretrain_human300)")
    for k in ("dataset_soup", "registry_commit", "filter_key"):
        _require(ident.get(k) == FINETUNE_LINEAGE[k],
                 f"dataset_identity.{k}={ident.get(k)!r} != {FINETUNE_LINEAGE[k]!r}")
    _require(bool(ident.get("evidence")), "dataset_identity.evidence 필요 (근거 없는 추정 금지)")
    _require(m.get("normalization_source") == "checkpoint_metadata",
             "action 정규화는 checkpoint metadata 통계만 허용 (데이터셋 재계산 금지)")
    _require(m.get("action_target") == "expert_dataset_action",
             "action target 은 데이터셋 expert action 이어야 함 (GR00T 예측으로 대체 금지)")
    split = _confirmed(m.get("split_policy"), "split_policy")
    _require(bool(split.get("decided_by")) and split.get("unit") in ("episode", "task", "scene"),
             "split_policy.decided_by/unit 필요")
    m["episodes"] = _check_episodes(m.get("episodes"), ALLOWED_TRAIN_SPLITS)
    m["membership_claim"] = _check_membership(m, purpose)
    return m


def _episode_parquet_rel(episode_index: int, info: Mapping[str, Any]) -> str:
    chunk = int(episode_index) // int(info["chunks_size"])
    return info["data_path"].format(episode_chunk=chunk, episode_index=int(episode_index))


def validate_aux_split(split_path: str | Path, *, data_root: str | Path | None = None,
                       verify_files: bool = False) -> dict:
    """메인이 만든 OpenDrawer_aux_split_v1.json 을 독립 재계산으로 검증한다.

    - 102 episode, id/index 유일, rank_hash = sha256(salt|episode_id), 최저 round(N·0.1) rank 가 val.
    - dataset_identity 의 info/episodes.jsonl sha 가 로컬 실측 상수와 같고, historical membership 을
      주장하지 않는다 ("unverified" 명시).
    - verify_files 면 data_root 아래 package 의 info.json·episodes.jsonl·parquet sha256 을 대조한다
      (경로는 파일에 적힌 절대경로가 아니라 data_root + info.json data_path 로 다시 만든다).
    반환: {episodes: [{episode_id, split, run_group}], counts, aux_split_sha256}
    """
    raw = Path(split_path).read_bytes()
    d = json.loads(raw)
    _require(d.get("status") == AUX_SPLIT_STATUS, f"aux split status {d.get('status')!r}")
    ident = d.get("dataset_identity") or {}
    _require(ident.get("info_sha256") == LOCAL_OPENDRAWER["info_sha256"] and
             ident.get("episodes_jsonl_sha256") == LOCAL_OPENDRAWER["episodes_jsonl_sha256"],
             "aux split dataset_identity sha 가 로컬 package 실측과 다름")
    _require(ident.get("package_episode_count") == LOCAL_OPENDRAWER["episodes"], "package episode 수 != 102")
    _require("unverified" in str(ident.get("historical_filter_membership", "")),
             "aux split 이 historical membership 을 확인됐다고 주장함 — 근거 없는 주장 거부")
    sp = d.get("split_policy") or {}
    _require(sp.get("unit") == "episode" and isinstance(sp.get("salt"), str), "split_policy.unit/salt")
    eps = d.get("episodes") or []
    n = len(eps)
    _require(n == LOCAL_OPENDRAWER["episodes"], f"episode {n} != 102")
    ids = [e["episode_id"] for e in eps]
    _require(len(set(ids)) == n and sorted(e["episode_index"] for e in eps) == list(range(n)),
             "episode_id/index 중복 또는 누락")
    for e in eps:
        h = hashlib.sha256(f"{sp['salt']}|{e['episode_id']}".encode()).hexdigest()
        _require(h == e.get("rank_hash"), f"{e['episode_id']}: rank_hash 재계산 불일치")
    n_val = int(round(n * 0.1))
    ranked = sorted(eps, key=lambda e: e["rank_hash"])
    want = {e["episode_id"]: ("val" if i < n_val else "train") for i, e in enumerate(ranked)}
    _require(all(e["split"] == want[e["episode_id"]] for e in eps), "split 이 salt 재계산과 다름")
    counts = {s: sum(e["split"] == s for e in eps) for s in ("train", "val", "test")}
    _require(counts == {k: int(v) for k, v in sp.get("counts", {}).items()} == {"train": n - n_val,
             "val": n_val, "test": 0}, f"split counts {counts}")
    if verify_files:
        if data_root is None:
            from scripts.path_setup import DATA_ROOT  # 경로 단일 출처

            data_root = DATA_ROOT
        pkg = Path(data_root) / OPENDRAWER_PACKAGE_REL
        _require(sha256_file(pkg / "meta" / "info.json") == LOCAL_OPENDRAWER["info_sha256"], "info.json sha")
        _require(sha256_file(pkg / "meta" / "episodes.jsonl") == LOCAL_OPENDRAWER["episodes_jsonl_sha256"],
                 "episodes.jsonl sha")
        info = json.loads((pkg / "meta" / "info.json").read_text())
        for e in eps:
            f = pkg / _episode_parquet_rel(e["episode_index"], info)
            _require(f.is_file() and sha256_file(f) == e.get("parquet_sha256"),
                     f"{e['episode_id']}: parquet sha256 불일치/없음")
    return {"episodes": [{"episode_id": e["episode_id"], "split": e["split"],
                          "run_group": e["episode_id"]} for e in eps],
            "counts": counts, "aux_split_sha256": hashlib.sha256(raw).hexdigest(),
            "files_verified": bool(verify_files)}


def package_smoke_manifest(split_path: str | Path, *, feature_extraction: Mapping[str, Any],
                           data_root: str | Path | None = None,
                           mode: str = "published_package_subset_for_smoke") -> dict:
    """published package subset manifest 를 조립한다 (feature_extraction 은 메인이 확정해 넘김).

    mode: published_package_subset_for_smoke | published_package_subset_for_baseline.
    """
    _require(mode in PACKAGE_SUBSET_PURPOSES, f"package manifest mode {mode!r} not in {PACKAGE_SUBSET_PURPOSES}")
    aux = validate_aux_split(split_path, data_root=data_root, verify_files=True)
    return {
        "contract_version": CONTRACT_VERSION,
        "checkpoint": {"config_sha256": CONFIG_SHA256, "metadata_sha256": METADATA_SHA256},
        "feature_extraction": dict(feature_extraction),
        "dataset_identity": {"status": "confirmed", "dataset_soup": FINETUNE_LINEAGE["dataset_soup"],
                             "registry_commit": FINETUNE_LINEAGE["registry_commit"],
                             "filter_key": FINETUNE_LINEAGE["filter_key"],
                             "evidence": ["outputs/analysis/checkpoint_training_data_20260916/report.md"],
                             "scope": "recipe identity only; OpenDrawer package subset"},
        "episode_membership": {"status": "confirmed", "mode": mode,
                               "historical_membership_claimed": False,
                               "package_inventory": {
                                   "info_sha256": LOCAL_OPENDRAWER["info_sha256"],
                                   "episodes_jsonl_sha256": LOCAL_OPENDRAWER["episodes_jsonl_sha256"],
                                   "package_episode_count": LOCAL_OPENDRAWER["episodes"]},
                               "aux_split_sha256": aux["aux_split_sha256"]},
        "normalization_source": "checkpoint_metadata",
        "action_target": "expert_dataset_action",
        "split_policy": {"status": "confirmed", "decided_by": "main session aux split (salted sha256)",
                         "unit": "episode"},
        "episodes": aux["episodes"],
    }


def _id_set(block: Mapping[str, Any], key: str) -> set:
    v = block.get(key) or []
    _require(isinstance(v, list), f"evaluation.{key} 는 list")
    return set(v)


def load_trqam_rollout_manifest(path: str | Path, *, feature_signature: str) -> dict:
    """(B) 우리 RoboCasa rollout → TRQAM 추가학습 manifest.

    episode 역할: historical_base / historical_expert (기존 조건) 와 changed_failure (바뀐 조건의
    실패, success is False). 바뀐 조건의 성공 또는 성공여부 미상은 adaptation 에서 거부한다.
    evaluation 블록의 episode/run_group/scenario_seed/noise_id/reset_id 는 학습 쪽과 서로소여야 한다.
    """
    m = json.loads(Path(path).read_text())
    _check_common(m, feature_signature)
    _confirmed(m.get("scenario"), "scenario")
    rc = _confirmed(m.get("reward_contract"), "reward_contract")
    _require(isinstance(rc.get("name"), str) and rc.get("name"), "reward_contract.name 필요")
    _require(isinstance(rc.get("gamma"), (int, float)) and 0 < rc["gamma"] <= 1,
             "reward_contract.gamma 필요")
    tr = m.get("transition") or {}
    _require(isinstance(tr.get("execute_steps"), int) and 1 <= tr["execute_steps"] <= ACTION_HORIZON,
             "transition.execute_steps (실제 실행 m, 1..16) 필요")
    eps = _check_episodes(m.get("episodes"), ALLOWED_TRAIN_SPLITS)
    for e in eps:
        eid = e["episode_id"]
        role = e.get("source_role")
        _require(role not in REJECTED_ROLES, f"{eid}: changed_success 는 adaptation 금지")
        _require(role in ADAPT_ROLES, f"{eid}: source_role {role!r} not in {ADAPT_ROLES}")
        _require(isinstance(e.get("success"), bool) or role.startswith("historical"),
                 f"{eid}: success 라벨 없음")
        if role == "changed_failure":
            _require(e.get("success") is False,
                     f"{eid}: changed 조건은 success is False 만 허용 (성공/미상 거부)")
        _require(e.get("end_reason") in ("terminal", "truncated"),
                 f"{eid}: end_reason 은 terminal|truncated 로 명시")
        _require(e.get("has_feature") is True, f"{eid}: decoder-입력 feature 캡처 없음")
        _require(isinstance(e.get("scenario_seed"), int), f"{eid}: scenario_seed 없음")
        check_capture_layout(e.get("feature_capture_point"), e.get("feature_axes"),
                             e.get("record_shape"), e.get("denoise_step_captured"))
    ev = _confirmed(m.get("evaluation"), "evaluation")
    pairs = {
        "episode_ids": {e["episode_id"] for e in eps},
        "run_groups": {e["run_group"] for e in eps},
        "scenario_seeds": {e["scenario_seed"] for e in eps},
        "noise_ids": {e["noise_id"] for e in eps if "noise_id" in e},
        "reset_ids": {e["reset_id"] for e in eps if "reset_id" in e},
    }
    _require(len(_id_set(ev, "episode_ids")) > 0 and len(_id_set(ev, "scenario_seeds")) > 0,
             "evaluation.episode_ids/scenario_seeds 비어 있음 (eval 고정 필수)")
    for key, train_vals in pairs.items():
        clash = sorted(map(str, _id_set(ev, key) & train_vals))
        _require(not clash, f"evaluation.{key} 가 adaptation 과 겹침: {clash[:5]}")
    return m


def check_capture_layout(capture_point: Any, axes: Any, shape: Any,
                         denoise_step_captured: Any = None) -> None:
    """record 가 decoder-입력 계약 (last16 평균 1024, denoise 0) 을 복원 가능한지 판정."""
    _require(isinstance(axes, list) and isinstance(shape, list) and len(axes) == len(shape),
             f"feature_axes/record_shape 누락 또는 길이 불일치: {axes} {shape}")
    _require("layer" not in axes and shape[-1] not in (DIT_WIDTH, VLM_CONTEXT_DIM),
             f"block residual/VLM capture {axes} {shape} 거부: action_decoder 입력 1024 와 다른 "
             "interface (대체 입력·차원 변환 금지)")
    _require(capture_point == FEATURE_CAPTURE_POINT,
             f"capture_point {capture_point!r} != {FEATURE_CAPTURE_POINT}")
    _require(axes[-1] == "feature_dim" and shape[-1] == FEATURE_DIM, f"feature_dim != 1024: {shape}")
    rest = axes[:-1]
    if "denoise_step" in rest:
        _require(rest[0] == "denoise_step", "denoise_step 축은 맨 앞")
        rest = rest[1:]
    else:
        _require(denoise_step_captured == 0,
                 "denoise_step 축 없는 record 는 denoise_step_captured=0 명시 필요 (K 평균 금지)")
    _require(rest in ([], ["action_token"], ["model_token"]),
             f"허용 layout 아님: {axes} (feature_dim | action_token | model_token)")
    if rest == ["action_token"]:
        _require(shape[axes.index("action_token")] == ACTION_HORIZON, "action_token 축 길이 != 16")
    if rest == ["model_token"]:
        _require(shape[axes.index("model_token")] == DIT_SEQ_LEN, "model_token 축 길이 != 49")


def feature_from_record(hidden: np.ndarray, *, axes: Sequence[str], capture_point: str,
                        denoise_step_captured: int | None = None) -> np.ndarray:
    """수집 record 1개 → QAM 관측 (1024,). denoise_step 0, 마지막 16 action token 평균."""
    h = np.asarray(hidden)
    axes = list(axes)
    _require(h.ndim == len(axes), f"record ndim {h.ndim} != axes {axes}")
    check_capture_layout(capture_point, axes, list(h.shape), denoise_step_captured)
    if axes[0] == "denoise_step":
        h, axes = h[0], axes[1:]
    if axes[0] == "model_token":
        h = h[-ACTION_HORIZON:].mean(axis=0)
    elif axes[0] == "action_token":
        h = h.mean(axis=0)
    out = np.asarray(h, dtype=np.float32)
    _require(out.shape == (FEATURE_DIM,), f"추출 결과 shape {out.shape}")
    _require(bool(np.all(np.isfinite(out))), "추출 feature 비유한")
    return out


def check_record_timing(env_steps: Sequence[int], execute_steps: int) -> None:
    """get_action 해상도: 0 시작, 정확히 execute_steps 간격 (chunk16 예측/execute 실행)."""
    s = np.asarray(env_steps)
    _require(1 <= execute_steps <= ACTION_HORIZON, f"execute_steps {execute_steps} not in [1,16]")
    _require(s.ndim == 1 and len(s) > 0 and s[0] == 0, "env_steps 는 0 부터 시작")
    _require(bool(np.all(np.diff(s) == execute_steps)),
             f"env_steps 간격 != {execute_steps} (feature/decision 시점 불일치)")
