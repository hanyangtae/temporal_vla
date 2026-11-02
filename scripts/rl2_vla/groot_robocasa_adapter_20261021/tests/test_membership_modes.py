"""episode membership 증거 모드: published package subset (smoke) vs original full reproduction."""
import copy
import hashlib
import json

import pytest

from conftest import REPO_ROOT
from data_contract import (
    LOCAL_OPENDRAWER,
    DataContractBlocked,
    load_finetune_data_manifest,
    package_smoke_manifest,
    validate_aux_split,
)
from test_data_contract import FULL, SIG, SMOKE, _write, finetune_manifest

AUX_SPLIT = REPO_ROOT / "outputs/analysis/groot_robocasa_lineage_20261021/OpenDrawer_aux_split_v1.json"
FE = {"status": "confirmed", "feature_signature": SIG,
      "capture_point": "action_head.action_decoder.input", "denoise_step": 0}


@pytest.fixture(scope="module")
def aux():
    if not AUX_SPLIT.is_file():
        pytest.skip("메인 aux split 파일 없음")
    return json.loads(AUX_SPLIT.read_text())


def _write_aux(tmp_path, d):
    p = tmp_path / "aux.json"
    p.write_text(json.dumps(d))
    return p


def test_real_aux_split_recomputes_and_checksums_match(aux):
    from scripts.path_setup import DATA_ROOT

    pkg = DATA_ROOT / "robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot"
    out = validate_aux_split(AUX_SPLIT, verify_files=pkg.is_dir())
    assert out["counts"] == {"train": 92, "val": 10, "test": 0}
    assert len(out["episodes"]) == 102
    assert out["aux_split_sha256"] == hashlib.sha256(AUX_SPLIT.read_bytes()).hexdigest()
    if not pkg.is_dir():
        pytest.skip("로컬 package 없음 — parquet checksum 대조 생략")
    assert out["files_verified"] is True


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["episodes"][0].update(split="val" if d["episodes"][0]["split"] == "train" else "train"),
     "split|counts"),
    (lambda d: d["episodes"][3].update(rank_hash="0" * 64), "rank_hash"),
    (lambda d: d["episodes"].pop(), "102"),
    (lambda d: d["dataset_identity"].update(historical_filter_membership="confirmed equal to 100_demos"),
     "historical"),
    (lambda d: d["dataset_identity"].update(info_sha256="0" * 64), "sha"),
    (lambda d: d.update(status="original_training_split"), "status"),
    (lambda d: d["split_policy"].update(salt="other"), "rank_hash"),
])
def test_aux_split_tampering_rejected(aux, tmp_path, mutate, match):
    d = copy.deepcopy(aux)
    mutate(d)
    with pytest.raises(DataContractBlocked, match=match):
        validate_aux_split(_write_aux(tmp_path, d))


def test_aux_split_count_preserving_swap_rejected(aux, tmp_path):
    """train 1개 ↔ val 1개 교환은 counts 가 같다 → salt 재계산만 잡을 수 있다."""
    d = copy.deepcopy(aux)
    tr = next(e for e in d["episodes"] if e["split"] == "train")
    va = next(e for e in d["episodes"] if e["split"] == "val")
    tr["split"], va["split"] = "val", "train"
    with pytest.raises(DataContractBlocked, match="salt 재계산"):
        validate_aux_split(_write_aux(tmp_path, d))


def test_aux_split_parquet_checksum_mismatch_rejected(aux, tmp_path):
    from scripts.path_setup import DATA_ROOT

    if not (DATA_ROOT / "robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot").is_dir():
        pytest.skip("로컬 package 없음")
    d = copy.deepcopy(aux)
    d["episodes"][5]["parquet_sha256"] = "f" * 64
    with pytest.raises(DataContractBlocked, match="parquet"):
        validate_aux_split(_write_aux(tmp_path, d), verify_files=True)


def test_package_smoke_manifest_validates_without_original_claim(aux, tmp_path):
    from scripts.path_setup import DATA_ROOT

    if not (DATA_ROOT / "robocasa/v1.0/pretrain/atomic/OpenDrawer/20250819/lerobot").is_dir():
        pytest.skip("로컬 package 없음")
    m = package_smoke_manifest(AUX_SPLIT, feature_extraction=FE)
    out = load_finetune_data_manifest(_write(tmp_path, m), feature_signature=SIG, purpose=SMOKE)
    claim = out["membership_claim"]
    assert claim["original_membership_confirmed"] is False
    assert "원래 100_demos membership 아님" in claim["data_identity_claim"]
    assert sum(e["split"] == "train" for e in out["episodes"]) == 92
    # 같은 manifest 로 original full reproduction 을 주장하면 차단 (historical 목록 미상).
    with pytest.raises(DataContractBlocked, match="historical 100_demos"):
        load_finetune_data_manifest(_write(tmp_path, m), feature_signature=SIG, purpose=FULL)


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m["episode_membership"].update(historical_membership_claimed=True), "historical_membership_claimed"),
    (lambda m: m["episode_membership"].pop("historical_membership_claimed"), "historical_membership_claimed"),
    (lambda m: m["episode_membership"].update(mode="historical_exact_100_demos"), "mode"),
    (lambda m: m["episode_membership"].pop("mode"), "mode"),
    (lambda m: m["episode_membership"]["package_inventory"].update(package_episode_count=100), "102"),
    (lambda m: m["episode_membership"]["package_inventory"].update(info_sha256="0" * 64), "info_sha256"),
    (lambda m: m["episode_membership"].pop("aux_split_sha256"), "aux_split"),
])
def test_smoke_mode_fail_closed(tmp_path, mutate, match):
    m = finetune_manifest()
    mutate(m)
    with pytest.raises(DataContractBlocked, match=match):
        load_finetune_data_manifest(_write(tmp_path, m), feature_signature=SIG, purpose=SMOKE)


def test_purpose_required_and_unknown_rejected(tmp_path):
    p = _write(tmp_path, finetune_manifest())
    with pytest.raises(TypeError):
        load_finetune_data_manifest(p, feature_signature=SIG)
    with pytest.raises(DataContractBlocked, match="purpose"):
        load_finetune_data_manifest(p, feature_signature=SIG, purpose="training")


def test_original_full_reproduction_requires_historical_list(tmp_path):
    m = finetune_manifest()
    with pytest.raises(DataContractBlocked, match="historical 100_demos"):
        load_finetune_data_manifest(_write(tmp_path, m), feature_signature=SIG, purpose=FULL)
    # 증거가 생기면 통과하는 경로가 있는지 (가상의 historical 목록; 실제 목록은 미상).
    ids = sorted(e["episode_id"] for e in m["episodes"])
    m["episode_membership"] = {"status": "confirmed", "mode": "historical_exact_100_demos",
                               "evidence": ["hypothetical"],
                               "historical_episode_list_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    out = load_finetune_data_manifest(_write(tmp_path, m), feature_signature=SIG, purpose=FULL)
    assert out["membership_claim"]["original_membership_confirmed"] is True
    m["episode_membership"]["historical_episode_list_sha256"] = "0" * 64
    with pytest.raises(DataContractBlocked, match="historical_episode_list_sha256"):
        load_finetune_data_manifest(_write(tmp_path, m), feature_signature=SIG, purpose=FULL)


def test_local_package_constants():
    assert LOCAL_OPENDRAWER["episodes_jsonl_sha256"].startswith("59b4d478")
