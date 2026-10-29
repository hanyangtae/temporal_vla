#!/usr/bin/env bash
# atomic65 per-task stager, executed on the storage host (승준) from its git checkout
# (code arrives only via `remote_compute.sh sync-code <branch>`), by atomic65_stage.py:
#   remote_compute.sh run "nice -n 10 bash scripts/rl2_vla/atomic65_stage_20261028/remote_task.sh <args>"
# The script refuses to run unless its own sha256 equals SCRIPT_SHA (= the local copy),
# so a stale remote checkout fails with CODE_MISMATCH instead of running old code.
# Tests run the same file locally (`bash remote_task.sh <args>`) against file:// URLs.
#
# Contract
#   - write-once: an archive, an extracted package or a receipt is never overwritten
#     or deleted. Our own leftovers (stale staging dir) are moved aside, not removed.
#   - every package episode is kept. The registry filter_key (100_demos) is recorded
#     as a derived membership list in the receipt, never applied by truncation.
#   - restartable: flock per task, curl -C - resume on <archive>.part, receipt last.
#   - every outcome prints exactly one "STATUS <NAME>" line; exit codes are the
#     STATUS_CODES table in atomic65_stage.py.
#
# Args: TASK REL URL EXPECTED_BYTES ROOT RESERVE_BYTES TASK_CAP_BYTES SCRIPT_SHA
#       FILTER_N FILTER_SEED ROOT_ALIAS
set -Eeuo pipefail

main() {
  local TASK="$1" REL="$2" URL="$3" EXPECTED="$4" ROOT="$5" RESERVE="$6" CAP="$7"
  local SCRIPT_SHA="$8" FILTER_N="$9" FILTER_SEED="${10}" ROOT_ALIAS="${11}"

  fail() { echo "STATUS $1"; echo "DETAIL ${3:-}"; exit "$2"; }
  trap 'echo "STATUS UNEXPECTED_ERROR"; echo "DETAIL line ${LINENO}: ${BASH_COMMAND}"; exit 70' ERR

  [[ "$TASK" =~ ^[A-Za-z0-9]+$ ]] || fail BAD_ARGS 64 "task=$TASK"
  [[ "$REL" =~ ^v1\.0/pretrain/atomic/${TASK}/[0-9]{8}/lerobot$ ]] || fail BAD_ARGS 64 "rel=$REL"
  [[ "$ROOT" == /* && "$ROOT" != *..* && "$ROOT" != *" "* ]] || fail BAD_ARGS 64 "root=$ROOT"
  [[ "$EXPECTED" =~ ^[1-9][0-9]*$ && "$RESERVE" =~ ^[0-9]+$ && "$CAP" =~ ^[1-9][0-9]*$ ]] \
    || fail BAD_ARGS 64 "sizes expected=$EXPECTED reserve=$RESERVE cap=$CAP"
  [[ "$URL" =~ ^(https://utexas\.box\.com/shared/static/[a-z0-9]+\.tar|file:///.+\.tar)$ ]] \
    || fail BAD_ARGS 64 "url=$URL"
  local self_sha; self_sha=$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1)
  [[ "$self_sha" == "$SCRIPT_SHA" ]] \
    || fail CODE_MISMATCH 66 "running $self_sha, expected $SCRIPT_SHA (remote checkout not synced?)"
  local tool
  for tool in curl flock tar sha256sum python3 df find sort xargs; do
    command -v "$tool" >/dev/null || fail MISSING_TOOL 65 "$tool"
  done

  local PKG="$ROOT/$REL"
  local ARCH="$ROOT/archives/${REL}.tar"          # mirrors box_links_ds key layout
  local PART="$ARCH.part"
  local RECEIPT="$ROOT/receipts/$TASK.json"
  local FILES="$ROOT/receipts/$TASK.files.sha256"
  local STAGE="$ROOT/staging/$TASK.extract"
  mkdir -p "$ROOT/receipts" "$ROOT/logs" "$ROOT/locks" "$ROOT/staging" \
           "$(dirname "$ARCH")" "$(dirname "$PKG")"
  exec > >(tee -a "$ROOT/logs/$TASK.log") 2>&1
  echo "BEGIN $(date -u +%FT%TZ) task=$TASK expected=$EXPECTED script_sha=$SCRIPT_SHA"

  exec 9>"$ROOT/locks/$TASK.lock"
  flock -n 9 || fail LOCKED 75 "another stager holds $TASK"

  if [[ -e "$RECEIPT" ]]; then
    [[ -d "$PKG" && -f "$ARCH" && -f "$FILES" ]] || fail RECEIPT_WITHOUT_DATA 12 "$RECEIPT"
    echo "RECEIPT_JSON $(tr -d '\n' < "$RECEIPT")"
    echo "STATUS ALREADY_COMPLETE"; exit 0
  fi
  (( EXPECTED <= CAP )) || fail OVER_TASK_CAP 3 "expected=$EXPECTED cap=$CAP"

  # leftovers of a crashed run of ours: move aside (never reuse, never delete)
  local stamp f; stamp=$(date -u +%Y%m%dT%H%M%SZ)
  for f in "$ARCH.sha256.tmp" "$FILES.tmp" "$RECEIPT.tmp" "$STAGE"; do
    if [[ -e "$f" ]]; then mv -n "$f" "$f.stale.$stamp"; echo "STALE_MOVED $f.stale.$stamp"; fi
  done

  avail() { df -B1 --output=avail "$ROOT" | tail -n1 | tr -d ' '; }
  gate() {  # <bytes still to be written>
    local need=$(( $1 + RESERVE )) have; have=$(avail)
    echo "GATE need_write=$1 reserve=$RESERVE avail=$have"
    (( have >= need )) || fail INSUFFICIENT_SPACE 5 "avail=$have need=$need"
  }

  # ---- archive: download/resume into .part, verify, then publish (no-clobber) ----
  if [[ ! -f "$ARCH" ]]; then
    local have_part=0
    if [[ -f "$PART" ]]; then have_part=$(stat -c %s "$PART"); fi
    (( have_part <= EXPECTED )) || fail PART_OVERSIZE 4 "part=$have_part expected=$EXPECTED"
    # archive remainder + extracted copy (~archive size) must fit above the reserve
    gate $(( EXPECTED - have_part + EXPECTED ))
    if (( have_part < EXPECTED )); then
      echo "DOWNLOAD resume_from=$have_part"
      local rc=0
      curl -fsSL --retry 5 --retry-delay 15 --retry-connrefused --connect-timeout 30 \
           --speed-limit 4096 --speed-time 300 -C - -o "$PART" "$URL" || rc=$?
      if (( rc == 33 )); then fail RANGE_UNSUPPORTED 6 "server refused resume; .part kept"; fi
      if (( rc != 0 )); then fail DOWNLOAD_FAILED 2 "curl rc=$rc; .part kept for resume"; fi
    fi
    local got; got=$(stat -c %s "$PART")
    (( got == EXPECTED )) || fail SIZE_MISMATCH 7 "got=$got expected=$EXPECTED; .part kept"
    check_members "$PART" || fail BAD_ARCHIVE 8 "unsafe or unexpected tar members"
    local sha; sha=$(sha256sum "$PART" | cut -d' ' -f1)
    echo "$sha  lerobot.tar" > "$ARCH.sha256.tmp"
    mv -n "$PART" "$ARCH"
    [[ -f "$ARCH" && ! -e "$PART" ]] || fail PUBLISH_RACE 9 "archive appeared concurrently"
    mv -n "$ARCH.sha256.tmp" "$ARCH.sha256"
  fi
  [[ -f "$ARCH.sha256" ]] || { echo "$(sha256sum "$ARCH" | cut -d' ' -f1)  lerobot.tar" > "$ARCH.sha256"; }
  local ARCH_SHA; ARCH_SHA=$(sha256sum "$ARCH" | cut -d' ' -f1)
  [[ "$ARCH_SHA" == "$(cut -d' ' -f1 "$ARCH.sha256")" ]] || fail ARCHIVE_HASH_MISMATCH 9 "$ARCH"
  local ARCH_BYTES; ARCH_BYTES=$(stat -c %s "$ARCH")
  (( ARCH_BYTES == EXPECTED )) || fail SIZE_MISMATCH 7 "archive=$ARCH_BYTES expected=$EXPECTED"
  check_members "$ARCH" || fail BAD_ARCHIVE 8 "unsafe or unexpected tar members"

  # ---- extract into staging; never extract over an existing package ----
  gate "$EXPECTED"
  local X="$STAGE/x"            # archive content; our own manifests live beside it in $STAGE
  mkdir -p "$X"
  tar -xf "$ARCH" -C "$X" --no-same-owner --no-same-permissions
  [[ -d "$X/lerobot" && ! -L "$X/lerobot" ]] || fail BAD_ARCHIVE 8 "no lerobot/ after extract"
  # the only allowed top-level extra is a regular README.md; the official downloader
  # extracts into the <date> dir, so it is mirrored there.
  local e EXTRAS=()
  while IFS= read -r -d '' e; do
    e=${e#"$X/"}
    [[ "$e" == lerobot ]] && continue
    [[ "$e" == README.md && -f "$X/$e" && ! -L "$X/$e" ]] \
      || fail BAD_ARCHIVE 8 "unexpected top-level member: $e"
    EXTRAS+=("$e")
  done < <(find "$X" -mindepth 1 -maxdepth 1 -print0)
  echo "EXTRAS ${EXTRAS[*]:-none}"
  manifest "$X/lerobot" > "$STAGE/files.sha256"

  local origin CHECK
  if [[ -e "$PKG" ]]; then
    # package already there (ours after a crash, or someone else's): adopt only if
    # byte-identical to the archive; otherwise leave it untouched and stop.
    manifest "$PKG" > "$STAGE/existing.sha256"
    cmp -s "$STAGE/files.sha256" "$STAGE/existing.sha256" \
      || fail PREEXISTING_MISMATCH 11 "$PKG differs from archive; untouched; see $STAGE"
    origin="preexisting_verified_identical_to_archive"; CHECK="$PKG"
  else
    origin="extracted_from_archive"; CHECK="$X/lerobot"
  fi
  local PDIR; PDIR=$(dirname "$PKG")
  for e in "${EXTRAS[@]}"; do   # same write-once rule for extras next to the package
    if [[ -e "$PDIR/$e" ]] && ! cmp -s "$X/$e" "$PDIR/$e"; then
      fail PREEXISTING_MISMATCH 11 "$PDIR/$e differs from archive; untouched"
    fi
  done
  mv -n "$STAGE/files.sha256" "$FILES.tmp"

  # ---- inventory + receipt, checked BEFORE publishing (relative paths only) ----
  if ! python3 - "$CHECK" "$FILES.tmp" "$RECEIPT.tmp" "$TASK" "$REL" "$URL" "$ARCH_SHA" \
      "$ARCH_BYTES" "$SCRIPT_SHA" "$FILTER_N" "$FILTER_SEED" "$ROOT_ALIAS" "$origin" \
      "$X" "${EXTRAS[@]}" <<'PY'
import hashlib, json, random, sys, time
from pathlib import Path
(pkg, files, out, task, rel, url, arch_sha, arch_bytes, script_sha,
 filter_n, filter_seed, root_alias, origin, xdir) = sys.argv[1:15]
extras = {e: hashlib.sha256((Path(xdir) / e).read_bytes()).hexdigest() for e in sys.argv[15:]}
pkg = Path(pkg); filter_n = int(filter_n); filter_seed = int(filter_seed)
info = json.loads((pkg / "meta/info.json").read_text())
eps = [json.loads(l) for l in (pkg / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
ids = [int(e["episode_index"]) for e in eps]
chunk = int(info.get("chunks_size", 1000))
video_keys = sorted(k for k, v in info.get("features", {}).items() if v.get("dtype") == "video")
missing_data, missing_video = [], []
for i in ids:
    c = i // chunk
    if not (pkg / info["data_path"].format(episode_chunk=c, episode_index=i)).is_file():
        missing_data.append(i)
    for vk in video_keys:
        if not (pkg / info["video_path"].format(episode_chunk=c, video_key=vk, episode_index=i)).is_file():
            missing_video.append([i, vk])
n_parquet = sum(1 for _ in (pkg / "data").rglob("*.parquet"))
frames = sum(int(e["length"]) for e in eps)
errors = []
if len(ids) != int(info["total_episodes"]): errors.append("episodes.jsonl rows != info.total_episodes")
if len(set(ids)) != len(ids): errors.append("duplicate episode_index")
if n_parquet != len(ids): errors.append(f"parquet files {n_parquet} != episodes {len(ids)}")
if frames != int(info["total_frames"]): errors.append("sum(length) != info.total_frames")
if missing_data: errors.append(f"missing parquet for {len(missing_data)} episodes")
if missing_video: errors.append(f"missing video for {len(missing_video)} (episode,key)")
if errors:
    print("INVENTORY_ERROR " + json.dumps(errors)); sys.exit(1)
# RoboCasa groot_utils.groot_dataset.get_subset_demos_filter_key: shuffle the
# episodes.jsonl order with random.seed(filter_key_seed) and keep the first N;
# N >= total means "all". Recorded only; the package keeps every episode.
order = list(ids); state = random.getstate(); random.seed(filter_seed)
random.shuffle(order); random.setstate(state)
member = sorted(order) if filter_n >= len(order) else sorted(order[:filter_n])
outside = sorted(set(ids) - set(member))
files_text = Path(files).read_text()
receipt = {
    "schema": "atomic65_stage_receipt_v1", "task": task, "status": "COMPLETE",
    "origin": origin, "root_alias": root_alias, "package_rel": rel,
    "archive_rel": "archives/" + rel + ".tar", "source_url": url,
    "archive_sha256": arch_sha, "archive_bytes": int(arch_bytes),
    "archive_extras_sha256": {str(Path(rel).parent / e): h for e, h in sorted(extras.items())},
    "package_files": len(files_text.splitlines()),
    "package_bytes": sum(p.stat().st_size for p in pkg.rglob("*") if p.is_file()),
    "package_manifest_sha256": hashlib.sha256(files_text.encode()).hexdigest(),
    "info_sha256": hashlib.sha256((pkg / "meta/info.json").read_bytes()).hexdigest(),
    "episodes_total": len(ids), "frames_total": frames,
    "video_keys": video_keys, "codebase_version": info.get("codebase_version"),
    "filter_key": f"{filter_n}_demos",
    "filter_rule": "robocasa groot_dataset.get_subset_demos_filter_key; seed=%d" % filter_seed,
    "filter_members_n": len(member), "episodes_outside_filter_n": len(outside),
    "filter_member_episode_indices": member,
    "episodes_outside_filter": outside,
    "filter_membership_note": "derived from package order with the registry rule; "
                              "historical training membership not verified; no episode removed",
    "stager_script_sha256": script_sha,
    "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}
Path(out).write_text(json.dumps(receipt, indent=1) + "\n")
PY
  then
    fail PACKAGE_INCOMPLETE 10 "inventory check failed (see INVENTORY_ERROR); not published"
  fi
  if [[ "$CHECK" != "$PKG" ]]; then
    mv -n -T "$X/lerobot" "$PKG"
    [[ ! -e "$X/lerobot" ]] || fail PUBLISH_RACE 9 "package appeared concurrently"
  fi
  for e in "${EXTRAS[@]}"; do
    if [[ ! -e "$PDIR/$e" ]]; then mv -n "$X/$e" "$PDIR/$e"; fi
    cmp -s "$X/$e" "$PDIR/$e" 2>/dev/null || [[ ! -e "$X/$e" ]] \
      || fail PUBLISH_RACE 9 "extra $e changed concurrently"
  done
  if [[ -e "$FILES" ]]; then   # crash between the two publishes below: must match
    cmp -s "$FILES.tmp" "$FILES" || fail FILE_MANIFEST_MISMATCH 13 "$FILES differs"
    mv -n "$FILES.tmp" "$FILES.tmp.dup.$stamp"
  else
    mv -n "$FILES.tmp" "$FILES"
  fi
  mv -n "$RECEIPT.tmp" "$RECEIPT"
  [[ ! -e "$RECEIPT.tmp" ]] || fail PUBLISH_RACE 9 "receipt appeared concurrently"
  echo "RECEIPT_JSON $(tr -d '\n' < "$RECEIPT")"
  echo "STATUS COMPLETE"
}

# tar members must be regular files or directories under lerobot/ (optionally
# ./lerobot/), plus the single top-level README.md that official archives carry
# (verified on AdjustWaterTemperature 2026-10-29); anything else top-level is
# rejected; never an absolute path or a .. component. Rejections print the first
# offending lines.
check_members() {
  local listing; listing=$(tar -tvf "$1") || return 1
  [[ -n "$listing" ]] || return 1
  if grep -qvE '^[-d]' <<<"$listing"; then
    echo "MEMBER_REJECT type: $(grep -vE '^[-d]' <<<"$listing" | head -3 | tr '\n' '|')"; return 1
  fi
  local names; names=$(tar -tf "$1") || return 1
  if grep -qvE '^(\./)?(lerobot(/|$)|README\.md$)' <<<"$names"; then
    echo "MEMBER_REJECT name: $(grep -vE '^(\./)?(lerobot(/|$)|README\.md$)' <<<"$names" | head -3 | tr '\n' '|')"
    return 1
  fi
  if grep -qE '(^|/)\.\.(/|$)' <<<"$names"; then return 1; fi
  return 0
}

# sorted, root-relative sha256 manifest of every file under a package dir
manifest() {
  (cd "$1" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum)
}

# no command here needs stdin; detach it so an ssh session cannot feed anything in
main "$@" </dev/null
exit $?
