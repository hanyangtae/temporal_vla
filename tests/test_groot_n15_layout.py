from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


# 2026-08-10 레포 검토(S1~S9) 완료판 트리로 재생성 — 검토 원장 docs/review/LEDGER.tsv 가 근거.
CANONICAL_GROOT_N15_ROBOCASA_FILES = {
    "scripts/safe/groot_n15/robocasa/README.md",
    # 2026-08-12 수집 세션(PR #85) 반입분 — 검토 완료판 이후 dev 머지로 추가.
    "scripts/safe/groot_n15/robocasa/analyze/dishwasher_rack_feasibility.py",
    "scripts/safe/groot_n15/robocasa/analyze/drawer_scene_feasibility.py",
    "scripts/safe/groot_n15/robocasa/analyze/ovenrack_feasibility.py",
    "scripts/safe/groot_n15/robocasa/collect/build_collection_plan.py",
    "scripts/safe/groot_n15/robocasa/collect/build_collection_plan_v1b.py",
    "scripts/safe/groot_n15/robocasa/analyze/instruction_pathway_features.py",
    "scripts/safe/groot_n15/robocasa/analyze/phase_separation.py",
    "scripts/safe/groot_n15/robocasa/analyze/summarize_instruction_cells.py",
    "scripts/safe/groot_n15/robocasa/analyze/summarize_instruction_seed_samples.py",
    "scripts/safe/groot_n15/robocasa/analyze/summarize_instruction_seed_scan.py",
    "scripts/safe/groot_n15/robocasa/analyze/wrong_grasp_vl_separation.py",
    "scripts/safe/groot_n15/robocasa/collect/HANDOFF_phase_event_collection.md",
    "scripts/safe/groot_n15/robocasa/collect/collect_grid.sh",
    "scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py",
    "scripts/safe/groot_n15/robocasa/collect/phase_live_render.py",
    "scripts/safe/groot_n15/robocasa/collect/ship_to_archive.sh",
    "scripts/safe/groot_n15/robocasa/eval/annotate_phase_video.py",
    "scripts/safe/groot_n15/robocasa/eval/env_step_gt_batch.py",
    "scripts/safe/groot_n15/robocasa/eval/internal_parity.py",
    "scripts/safe/groot_n15/robocasa/eval/lerobot_http_eval.py",
    "scripts/safe/groot_n15/robocasa/eval/native_official_zmq_eval.py",
    "scripts/safe/groot_n15/robocasa/eval/native_zmq_eval.py",
    "scripts/safe/groot_n15/robocasa/eval/rejudge_success.py",
    "scripts/safe/groot_n15/robocasa/run_config.py",
    "scripts/safe/groot_n15/robocasa/run_config.sh",
    "scripts/safe/groot_n15/robocasa/split/build_safe_splits.py",
    "scripts/safe/groot_n15/robocasa/split/merge_seen60_source.py",
    "scripts/safe/groot_n15/robocasa/split/prepare_seen5_trainval_cp_test_split.py",
    "scripts/safe/groot_n15/robocasa/steer/exp3/README.md",
    "scripts/safe/groot_n15/robocasa/steer/exp4_1/build_t0_manifest.py",
    "scripts/safe/groot_n15/robocasa/steer/exp4_1/cells.env",
    "scripts/safe/groot_n15/robocasa/steer/exp4_1/conceptor_layer_sweep.py",
    "scripts/safe/groot_n15/robocasa/steer/exp4_1/diag_st_significance.py",
    "scripts/safe/groot_n15/robocasa/steer/exp4_1/diag_token_space.py",
    "scripts/safe/groot_n15/robocasa/steer/exp5_3/aggregate_ws_steer.py",
    "scripts/safe/groot_n15/robocasa/steer/exp5_4/seed_manifest.tsv",
    "scripts/safe/groot_n15/robocasa/steer/patchceil/PROTOCOL.md",
    "scripts/safe/groot_n15/robocasa/utils/prepare_base_new_embodiment.py",
    "scripts/safe/groot_n15/robocasa/utils/runtime.py",
    "scripts/safe/groot_n15/robocasa/vis/annotate_phase_video.py",
}

REMOVED_GROOT_N15_ROBOCASA_FILES = {
    "scripts/eval/groot_n15_official_robocasa_eval.py",
    "scripts/eval/groot_n15_robocasa_zmq_eval.py",
    "scripts/eval/lerobot_groot_n15_internal_parity.py",
    "scripts/eval/lerobot_groot_n15_official_robocasa_eval.py",
    "scripts/eval/run_groot_n15_target15_seedpairs.sh",
    "scripts/data/build_safe_groot_n15_splits.py",
    "scripts/data/merge_safe_groot_n15_seen60_source.py",
    "scripts/utils/lerobot_groot_n15_runtime.py",
    "scripts/utils/prepare_groot_n15_base_new_embodiment.py",
}

DISALLOWED_GROOT_N15_BACKEND_LIBRARY_PARTS = {
    "serve",
    "features.py",
    "loader.py",
    "schema.py",
    "io.py",
}


def test_groot_n15_robocasa_files_live_under_safe_tree() -> None:
    root = REPO_ROOT / "scripts/safe/groot_n15/robocasa"
    source_files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if "__pycache__" in relative_parts or path.suffix == ".pyc":
            continue
        source_files.append(str(path.relative_to(REPO_ROOT)))

    assert sorted(source_files) == sorted(CANONICAL_GROOT_N15_ROBOCASA_FILES)


def test_groot_n15_bundle_has_top_level_entrypoint_readme() -> None:
    assert (REPO_ROOT / "scripts/safe/groot_n15/README.md").is_file()


def test_old_groot_n15_robocasa_script_paths_are_removed() -> None:
    remaining = [
        path
        for path in sorted(REMOVED_GROOT_N15_ROBOCASA_FILES)
        if (REPO_ROOT / path).exists()
    ]
    assert remaining == []


def test_groot_n15_tree_does_not_grow_backend_library_parts() -> None:
    root = REPO_ROOT / "scripts/safe/groot_n15/robocasa"
    unexpected = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if DISALLOWED_GROOT_N15_BACKEND_LIBRARY_PARTS.intersection(relative_parts):
            unexpected.append(str(path.relative_to(REPO_ROOT)))
    assert unexpected == []
