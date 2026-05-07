---
name: Phase 1 training approaches — window vs full trajectory
description: Two Phase 1 training approaches tried (window-based current, full trajectory VITA-aligned), their differences in TTT memory accumulation and train/test mismatch
type: project
---

Window-based approach is the current implementation; full trajectory approach was tried and reverted (commit 5abc395, 2026-04-02).

**Why:** Full trajectory approach matches VITA paper exactly but was reverted. Window approach was the original implementation and has evaluation results (Pearson r=0.71 val, 0.65 unseen at step 50k). The key concern documented in docs/phase1_status.md is train/test mismatch — window-based training only accumulates 8 frames of TTT memory, while inference processes entire episodes sequentially.

**How to apply:** When discussing training strategy or debugging poor monotonicity/generalization, this train/test mismatch is the primary suspected cause. The revert commit message provides details of the structural changes needed (episode-level samples, pred_mask for dissimilarity-selected frames only, padding to 120).
