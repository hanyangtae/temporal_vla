"""Shared paths and run identity for GR00T N1.6 RoboCasa SAFE wiring.

Run identity is read from the same ``ROBOCASA_SAFE_*`` environment variables
that ``run_config.sh`` honors, so a single env file (e.g. ``seen18_env.sh``)
drives both the shell training recipes and the Python analysis scripts. The
defaults reproduce the original seen4/unseen2 run when no env is set.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
WORKSPACE_ROOT = REPO_ROOT.parent
SAFE_ROOT = Path(os.environ.get("SAFE_REPO", str(WORKSPACE_ROOT / "SAFE")))

OUT_ROOT = REPO_ROOT / "outputs/eval/robocasa/groot_n16"
RUN_ID = os.environ.get("ROBOCASA_SAFE_RUN_ID", "safe_seen4_unseen2_100ep")
RUN_ROOT = OUT_ROOT / RUN_ID

SPLIT_ROOT = RUN_ROOT / "split"
ANALYSIS_ROOT = RUN_ROOT / "analysis"
FEATURE_SPACE_ROOT = RUN_ROOT / "visualizations/feature_space"
CONFORMAL_FIGURE_ROOT = RUN_ROOT / "visualizations/conformal_figure"
FINAL_DETECTOR_DIR = RUN_ROOT / "final_detector"

EXPERIMENT_ID = os.environ.get(
    "ROBOCASA_SAFE_EXPERIMENT_ID", "seen4_unseen2_openDrawer_pnpCab_100ep"
)
SAFE_SUBSET_NAME = os.environ.get("ROBOCASA_SAFE_SUBSET_NAME", f"robocasa_{EXPERIMENT_ID}")
EXPERIMENT_FEATURE_SPACE_ROOT = FEATURE_SPACE_ROOT / EXPERIMENT_ID

FINAL_HORIZON_IDX_REL = os.environ.get("ROBOCASA_SAFE_FINAL_HORIZON_IDX_REL", "mean")
FINAL_DIFF_IDX_REL = os.environ.get("ROBOCASA_SAFE_FINAL_DIFF_IDX_REL", "concat-2")
FINAL_AGGREGATION_SLUG = os.environ.get("ROBOCASA_SAFE_FINAL_AGGREGATION_SLUG", "hmean_dconcat2")
FINAL_LR = os.environ.get("ROBOCASA_SAFE_FINAL_LR", "3e-4")
FINAL_LAMBDA_REG = os.environ.get("ROBOCASA_SAFE_FINAL_LAMBDA_REG", "1")
FINAL_SEED = int(os.environ.get("ROBOCASA_SAFE_FINAL_SEED", "2"))
FINAL_TRAIN_SUFFIX = os.environ.get(
    "ROBOCASA_SAFE_FINAL_TRAIN_SUFFIX", "agg_hmean_dconcat_2_lr3e_4_reg1_seed2"
)

HPARAM_SWEEP_ID = os.environ.get(
    "ROBOCASA_SAFE_HPARAM_SWEEP_ID", f"hparam_sweep_{FINAL_AGGREGATION_SLUG}_v2"
)
HPARAM_SWEEP_ROOT = RUN_ROOT / "experiments" / HPARAM_SWEEP_ID
HPARAM_SWEEP_LOG_ROOT = HPARAM_SWEEP_ROOT / "train_logs"
HPARAM_SWEEP_REPORT_ROOT = HPARAM_SWEEP_ROOT / "reports"
HPARAM_SWEEP_WANDB_ROOT = HPARAM_SWEEP_ROOT / "wandb"
HPARAM_SWEEP_HYDRA_ROOT = HPARAM_SWEEP_ROOT / "hydra"

FINAL_LSTM_RUN_NAME = (
    f"groot_n16-{SAFE_SUBSET_NAME}-lstm-{FINAL_TRAIN_SUFFIX}"
)
FINAL_LSTM_RUN_DIR = HPARAM_SWEEP_LOG_ROOT / FINAL_LSTM_RUN_NAME / "20260520" / "225235"

AGGREGATION_ABLATION_ROOT = RUN_ROOT / "experiments/aggregation_ablation"
AGGREGATION_ABLATION_LOG_ROOT = AGGREGATION_ABLATION_ROOT / "train_logs"
AGGREGATION_ABLATION_REPORT_ROOT = AGGREGATION_ABLATION_ROOT / "reports"
