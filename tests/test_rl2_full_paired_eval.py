import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rl2_vla/stage2_cluster_reward'))
from run_full_paired_eval import ARMS, TASKS, missing_batches, expected_keys


def test_reuse_and_missing_exactly_partition_four_arm_evaluation():
    found={a:expected_keys(a,reused=True) for a in ARMS}
    assert sum(map(len,found.values()))==296
    new=0
    for _,arms,seeds,start,trials in missing_batches():
        for a in arms:
            keys={f'{t}/env{e}/policy{s}' for t in TASKS for s in seeds for e in range(start,start+trials)}
            assert not (keys & found[a])
            found[a]|=keys;new+=len(keys)
    assert new==904
    assert all(found[a]==expected_keys(a) and len(found[a])==300 for a in ARMS)
