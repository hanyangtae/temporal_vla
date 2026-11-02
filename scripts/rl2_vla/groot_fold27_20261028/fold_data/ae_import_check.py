"""CPU import check of the AE potential source closure on a deploy (run from the deploy root, JAX_PLATFORMS=cpu):
finetune_fold_v2 -> ae_potential -> prior ae_cluster_reward/{tc_import,potential,cache_io} -> stage2_cluster_reward/{data,clusters},
grid_phase/ae_cluster.py standardizer, manuscript/numbers.md, task_classification phase modules at the pinned 33e5e04."""
import sys
sys.path[:0] = ['scripts/rl2_vla/groot_fold27_20261028/trqam', 'scripts/rl2_vla/groot_fold27_20261028']
import ae_potential as AP
import finetune_fold_v2  # noqa: F401
tc, pot = AP.prior_modules()
import cache_io
sys.path.insert(0, str(cache_io.STAGE2))
import data, clusters  # noqa: E401  stage2_cluster_reward
std = tc.load_manuscript_standardizer()
T = tc.TaskClassification(AP.TC_ROOT)
assert AP.tc_head() == AP.EXPECT_TC, AP.tc_head()
print('AE_IMPORT_OK', AP.PRIOR.relative_to(AP.ROOT), tc.MANUSCRIPT_NUMBERS.is_file(), data.__file__.split('scripts/')[1], clusters.__file__.split('scripts/')[1], type(T).__name__, sorted(std)[:2] if isinstance(std, dict) else type(std).__name__)
