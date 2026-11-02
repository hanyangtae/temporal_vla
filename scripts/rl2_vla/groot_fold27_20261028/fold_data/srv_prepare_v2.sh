#!/usr/bin/env bash
# srv48 / srv50 fold27 preparation, v2. Supersedes srv_prepare.sh for the steps below (srv_prepare.sh push-deps must NOT
# run: it moved source via push-data). Runs FROM kanu; remote actions = `remote_compute.sh run` in the deploy dir.
#
# Git layout (verified 2026-11-02, both hosts): the deploy /tmp/temporal-vla-fold27-production-20261102 is a git WORKTREE of
# the original /home/junhyeong/pkt_ws/temporal_vla (common dir .../temporal_vla/.git). Therefore:
#   * objects fetched/pushed here and the ref refs/fold27/pin/trqam_retention are written into that SHARED object store/refs;
#   * `git worktree add` registers metadata under the shared .git/worktrees/;
#   * the original checkout's HEAD, branch, index and working tree are NOT changed (tests assert this on a replica).
# Source moves ONLY through git. AE recipe closure (ae_potential.PRIOR = <deploy>/.worktrees/trqam-retention/...):
#   scripts/rl2_vla/ae_cluster_reward/{potential,cache_io,tc_import}.py -> scripts/rl2_vla/stage2_cluster_reward/{data,clusters}.py
#   + scripts/analysis/grid_phase/ae_cluster.py (standardizer) + manuscript/numbers.md (provenance), at commit 8070d4c5
#   (tc_import records this worktree's HEAD as this_repo_git, so the exact commit is required, not a content-equal one).
#   Commit presence 2026-11-02: srv50 shared repo HAS 8070d4c5 (no transfer); srv48 does not -> one of:
#     ae-pack-rc        git pack of exactly 8070d4c5 ^f76d19a5 streamed through `remote_compute.sh run` stdin into
#                       git index-pack (recommended: canonical wrapper, git objects only, nothing published, no file copy)
#     ae-fetch-origin   git fetch origin <sha>   (only after a user-approved publish of the exact commit)
#     ae-fetch-bundle   git fetch from the verified bundle git_bundles/trqam_retention.bundle (if the bundle file is placed)
#   then ae-worktree: `git worktree add --detach .worktrees/trqam-retention <sha>` + blob-by-blob check of every closure file.
# No QAM/AE source push-data fallback exists here.
#
# usage: bash srv_prepare_v2.sh <srv48|srv50> <check|ae-pack-rc|ae-fetch-origin|ae-fetch-bundle|ae-worktree|ae-import|verify|eval-inventory> [container]
set -euo pipefail
cd "$(dirname "$0")/../../../.."
HOST=${1:?srv48|srv50}; STEP=${2:?step}; CONTAINER=${3:-}
DEPLOY=/tmp/temporal-vla-fold27-production-20261102
O=outputs/analysis/groot_fold27_20261028
AE_SHA=8070d4c5a581e5734304c763a1c1684de5d957cd
AE_PREREQ=f76d19a537b919ce2eff7b5e15573f687e31b3c9                  # merge-base(AE_SHA, deploy 04092cbf) = bundle prerequisite
AE_WT=.worktrees/trqam-retention
AE_PATHS="scripts/rl2_vla/ae_cluster_reward scripts/rl2_vla/stage2_cluster_reward scripts/analysis/grid_phase/ae_cluster.py manuscript/numbers.md"
# deploy gitlinks (c354946c == 04092cbf gitlinks): path=sha
PINS="RL2-VLA=1fdccf350bfd9bfd003f99aaee2178009eb649ca lerobot=1396b9fab7aecddd10006c33c47a487ffdcb54b4
src/benchmarks/robocasa=3a6b40d01be7acd7511c893c6ea034f8ab940d69 src/benchmarks/robosuite=aaa8b9b214ce8e77e82926d677b4d61d55e577ab
src/policies/Isaac-GR00T=108f3381d0b4ac6dd79dfd1c699734b230d03931 task_classification=33e5e0495d8918b19e87da105bff735d2d80461d"
case "$HOST" in
  srv50) IP=166.104.35.50
         EPDIRS="srv50_bread_s4 srv50_candle_s0 srv50_jug_s1 srv50_marshmallow_s4"
         COMPACT="$O/compact/bread_s4 $O/compact/remaining_b_v3" ;;
  srv48) IP=166.104.35.48
         EPDIRS="srv48_dishrack_left_s3 srv48_dishrack_right_s1 srv48_ovenrack_left_s2"
         COMPACT="outputs/rl2_vla/fold27_mil_20261030_continuation/compact_verified" ;;
  *) echo "unknown host $HOST" >&2; exit 2 ;;
esac
RC() { REMOTE_USER=junhyeong REMOTE_HOST=$IP REMOTE_PORT=22 REMOTE_REPO=$DEPLOY bash scripts/utils/remote_compute.sh run "$1"; }
# tests only: run the identical remote command strings inside a local throwaway clone instead of over ssh
if [ -n "${SRV_PREPARE_LOCAL_DEPLOY:-}" ]; then RC() { (cd "$SRV_PREPARE_LOCAL_DEPLOY" && bash -c "$1"); }; fi

# episode records = <id>.json with a sibling <id>.npz (contract.json / COMPLETE.json / summaries are not records)
COUNT='n=0; for j in "$d"/*.json; do b=$(basename "$j" .json); [ "$b" = contract ] || [ "$b" = COMPLETE ] && continue; [ -f "$d/$b.npz" ] && n=$((n+1)); done'

case "$STEP" in
check)   # read-only
  RC "echo common=\$(git rev-parse --path-format=absolute --git-common-dir) gitdir=\$(git rev-parse --path-format=absolute --git-dir);
      echo HEAD \$(git rev-parse HEAD) shallow=\$(git rev-parse --is-shallow-repository);
      for kv in $(echo $PINS); do p=\${kv%%=*}; want=\${kv#*=}; got=\$(git -C \$p rev-parse HEAD 2>/dev/null || echo none);
        d=\$(git -C \$p status --porcelain --untracked-files=no 2>/dev/null | wc -l); echo \"pin \$p \$([ \$got = \$want ] && echo OK || echo MISMATCH:\$got) tracked_dirty=\$d\"; done;
      git cat-file -e $AE_PREREQ^{commit} && echo AE_PREREQ_PRESENT || echo AE_PREREQ_MISSING;
      git cat-file -e $AE_SHA^{commit} 2>/dev/null && echo AE_COMMIT_PRESENT || echo AE_COMMIT_ABSENT;
      [ -e $AE_WT ] && echo AE_WORKTREE \$(git -C $AE_WT rev-parse HEAD) || echo AE_WORKTREE_ABSENT;
      ls -d .fold27_env/bin/python 2>&1; df -h /tmp | tail -1" ;;
ae-pack-rc)
  # git objects through the canonical wrapper: remote_compute.sh run forwards stdin over its ssh, so the exact commit's
  # missing objects (pack of 8070d4c5 ^f76d19a5, thin against the prerequisite the deploy already has) stream into
  # `git index-pack --stdin --fix-thin` in the deploy's object store; then the pin ref. No file copy, nothing published,
  # no unrelated history (rev range is exact), connectivity re-checked by rev-list before the ref is set.
  git cat-file -e "$AE_PREREQ^{commit}"
  printf '%s\n^%s\n' "$AE_SHA" "$AE_PREREQ" | git pack-objects --revs --thin --stdout -q | \
    RC "set -e; git cat-file -e $AE_PREREQ^{commit}; git index-pack --stdin --fix-thin >/dev/null;
        git rev-list --objects $AE_SHA ^$AE_PREREQ >/dev/null; git update-ref refs/fold27/pin/trqam_retention $AE_SHA;
        git cat-file -e $AE_SHA^{commit} && echo AE_COMMIT_FETCHED"
  ;;
ae-fetch-origin)
  RC "set -e; git fetch --no-tags origin $AE_SHA && git cat-file -e $AE_SHA^{commit} && echo AE_COMMIT_FETCHED" ;;
ae-fetch-bundle)
  B=$O/git_bundles/trqam_retention.bundle
  WANT=$(python3 -c "import json;print(json.load(open('$O/git_bundles/BUNDLES.json'))['bundles']['trqam_retention']['sha256'])")
  RC "set -e; test \"\$(sha256sum $B | cut -d' ' -f1)\" = $WANT || { echo BUNDLE_SHA_MISMATCH; exit 3; };
      git bundle verify $B >/dev/null && test \"\$(git bundle list-heads $B | cut -d' ' -f1)\" = $AE_SHA &&
      git fetch --no-tags $B refs/fold27/pin/trqam_retention:refs/fold27/pin/trqam_retention && git cat-file -e $AE_SHA^{commit} && echo AE_COMMIT_FETCHED" ;;
ae-worktree)
  # expected blobs from the LOCAL commit (metadata only), compared with what the deploy worktree checks out
  EXP=$(git ls-tree -r $AE_SHA -- $AE_PATHS | awk '$4 ~ /\.(py|md)$/ {print $3"="$4}' | tr '\n' ' ')
  N=$(echo $EXP | wc -w)
  RC "set -e; git cat-file -e $AE_SHA^{commit};
      [ -e $AE_WT ] || git worktree add --detach $AE_WT $AE_SHA;
      test \"\$(git -C $AE_WT rev-parse HEAD)\" = $AE_SHA; test -z \"\$(git -C $AE_WT status --porcelain --untracked-files=no -- $AE_PATHS)\";
      ok=0; for kv in $EXP; do b=\${kv%%=*}; f=\${kv#*=}; [ \"\$(git hash-object $AE_WT/\$f)\" = \$b ] && ok=\$((ok+1)) || echo BLOB_MISMATCH \$f; done;
      echo AE_CLOSURE_BLOBS_OK \$ok/$N; test \$ok = $N" ;;
ae-import)
  # CPU import of the exact chain finetune_fold_v2 -> ae_potential -> prior tc_import/potential/cache_io(+stage2 data) + TC phase modules
  RC "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= .fold27_env/bin/python scripts/rl2_vla/groot_fold27_20261028/fold_data/ae_import_check.py" ;;
verify)   # 200 episode records per staged scene (json+npz pairs, contract excluded) + compact counts
  for name in $EPDIRS; do
    RC "d=$O/full_episodes/$name; $COUNT; echo $name records=\$n contract=\$(sha256sum \$d/contract.json | cut -c1-16); test \$n = 200"
  done
  for c in $COMPACT; do
    RC "d=$c; $COUNT; echo $c records=\$n contract=\$(sha256sum \$d/contract.json 2>/dev/null | cut -c1-16)"
  done ;;
eval-inventory)   # read-only: deploy runtime paths + the env container the isolated eval would exec into
  RC "set -e; for p in scripts/rl2_vla/groot_fold27_20261028/eval/runtime scripts/safe/groot_n15/robocasa/collect/http_feature_collect.py \
        scripts/serve/lerobot.py configs/checkpoints/lerobot_groot_n15__robocasa365_ckpt120000.yaml scripts/rl2_vla/groot_robocasa_adapter_20261021 \
        src/benchmarks/robocasa/robocasa src/benchmarks/robosuite/robosuite src/policies/Isaac-GR00T/gr00t lerobot/src $O/srv_cache_bundle; do
        [ -e \$p ] && echo \"present \$p\" || echo \"MISSING \$p\"; done;
      ls -d ~/miniconda3/envs/lerobot_050_groot/bin/python 2>&1;
      docker ps --format '{{.Names}}|{{.Image}}' | grep -i robocasa || true;
      if [ -n '$CONTAINER' ]; then docker inspect '$CONTAINER' --format '{{json .Mounts}}|{{.Image}}|{{.State.Running}}' | python3 -c '
import json,sys,os; m,img,run=sys.stdin.read().strip().split(\"|\"); ms=json.loads(m)
hit=[x for x in ms if x[\"Destination\"]==\"/temporal_vla\"]
print(\"container\",\"$CONTAINER\",\"image\",img,\"running\",run)
for x in hit: print(\"mount /temporal_vla <-\",x[\"Source\"],\"RW\",x.get(\"RW\"),\"DEPLOY_MATCH\",os.path.realpath(x[\"Source\"])==os.path.realpath(\"$DEPLOY\"))'; fi" ;;
*) echo "unknown step $STEP" >&2; exit 2 ;;
esac
