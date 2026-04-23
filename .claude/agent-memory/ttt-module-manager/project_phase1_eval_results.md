---
name: Phase 1 evaluation results and hyperparameters
description: Phase 1 window-based training evaluation metrics and key hyperparameter values used in practice
type: project
---

Evaluation results from window-based training (100k steps, stride=4, Adam — old setting):

| Split  | ckpt     | MSE    | MAE    | Pearson r | Mono rate |
|--------|----------|--------|--------|-----------|-----------|
| train  | step 50k | 0.0428 | 0.1603 | 0.858     | 0.600     |
| val    | step 50k | 0.0617 | 0.1969 | 0.713     | 0.584     |
| unseen | step 50k | 0.0685 | 0.2073 | 0.654     | 0.569     |

Step 50k is better than 100k for val/unseen (overfitting). Mono rate ~0.58 is weak.

Key hyperparameters: input_dim=1024, proj_dim=64, inner_model_type=mlp, eta_base=0.1, head_hidden_dim=128, lambda_self=0.5, lr=1e-4, batch_size=32, ~210K params total.

**Why:** These are baselines for comparison with VITA-aligned re-training (stride=1, AdamW+cosine LR, epoch-based).

**How to apply:** Use step 50k as Phase 2 starting point. Re-training with corrected settings (docs/phase1_status.md modifications) has been prepared but not executed as of 2026-04-01.
