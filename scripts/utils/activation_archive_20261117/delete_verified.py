#!/usr/bin/env python3
"""Delete source files whose bytes provably exist at Seungjun (fold27 activation cleanup).

DRY-RUN BY DEFAULT. Deletes only with --execute AND --plan-sha256 matching the reviewed plan.
  --mode identical    klass IDENTICAL; evidence = fresh sha256sum of the existing Seungjun copy (--dst-sha)
  --mode transferred  klass MISSING/DIFFERENT/SHA_ONLY; evidence = verify_archive.py output made at the
                      destination (--verified); rows not in it are not touched
Per file (host == --host), immediately before unlink:
  1 path has no symlink component, no '..'; is regular, nlink==1
  2 size / mtime_ns / inode equal the plan pin
  3 fresh sha256 == plan sha256 == destination evidence
  4 no process holds it open (fd/cwd/mmap scan of /proc, refreshed every --rescan files and again
    right before each unlink in --execute). A held file is EXCLUDED (SKIP_REFERENCED, kept on disk);
    unreadable same-uid fds (except sshd/sd-pam) or a vanished --protect-pid -> abort
  5 re-lstat right before unlink (TOCTOU narrow) -> unlink
Any failure aborts the whole run (fail-closed); already-deleted files are logged, so a rerun with the
same plan reports them as GONE and continues with the rest only if --allow-gone.
Directories are never removed. Log TSV (write-once) records every action.

  python3 delete_verified.py --plan plan.tsv --plan-sha256 <sha> --host 50 \
      --base /tmp/temporal-vla-fold27-production-20261102/outputs/analysis/groot_fold27_20261028 \
      --mode identical --dst-sha sj_sha_recheck.txt \
      --protect-pid 558741 --protect-pid 1238295 --log del50.tsv [--execute]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cleanup_common as C  # noqa: E402


def read_dst_sha(path):
    out = {}
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln:
                d, rel = ln.split("  ", 1)
                out[rel] = d
    return out


def read_verified(paths):
    out = {}
    for p in paths:
        with open(p) as f:
            lines = f.read().splitlines()
        if not lines or lines[0] != "relpath\tsize\tsha256" or any(x.startswith("# FAILED") for x in lines):
            raise SystemExit(f"{p}: not a successful verify_archive.py output")
        for ln in lines[1:]:
            rel, size, sha = ln.split("\t")
            if rel in out:
                raise SystemExit(f"{rel} verified twice")
            out[rel] = (int(size), sha)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--plan-sha256", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--mode", choices=("identical", "transferred"), required=True)
    ap.add_argument("--dst-sha", help="identical mode: fresh sha256sum of Seungjun copies (relpaths = plan dst_match)")
    ap.add_argument("--verified", action="append", default=[],
                    help="transferred mode: verify_archive.py output(s) produced AT THE DESTINATION")
    ap.add_argument("--protect-pid", action="append", default=[])
    ap.add_argument("--allow-unreadable-comm", action="append", default=[],
                    help="reviewed same-uid process names whose fds may be unreadable (sshd,(sd-pam) built in)")
    ap.add_argument("--log", required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--allow-gone", action="store_true")
    ap.add_argument("--rescan", type=int, default=50)
    ap.add_argument("--proc", default="/proc", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    rows, _ = C.read_plan(a.plan, a.plan_sha256)
    if a.mode == "identical":
        if not a.dst_sha or a.verified:
            raise SystemExit("identical mode needs --dst-sha (and no --verified)")
        todo = [r for r in rows if r["host"] == a.host and r["klass"] == "IDENTICAL"]
        dst = read_dst_sha(a.dst_sha)

        def evidence_ok(r):
            return dst.get(r["dst_match"]) == r["sha256"]
    else:
        if not a.verified or a.dst_sha:
            raise SystemExit("transferred mode needs --verified (and no --dst-sha)")
        todo = [r for r in rows if r["host"] == a.host and r["klass"] in ("MISSING", "DIFFERENT", "SHA_ONLY")]
        ver = read_verified(a.verified)
        todo = [r for r in todo if r["relpath"] in ver]  # only members actually packed+restore-verified

        def evidence_ok(r):
            return ver[r["relpath"]] == (int(r["size"]), r["sha256"])
    if os.path.exists(a.log):
        raise SystemExit(f"{a.log} exists; logs are write-once")
    mode = "EXECUTE" if a.execute else "DRYRUN"
    log = open(a.log, "x")
    log.write("ts\tmode\taction\trelpath\tsize\tsha256\tdetail\n")

    def emit(action, r, detail=""):
        log.write(f"{time.time():.3f}\t{mode}\t{action}\t{r['relpath']}\t{r['size']}\t{r['sha256']}\t{detail}\n")
        log.flush()

    done_bytes, n, skipped = 0, 0, 0
    try:
        # Destination check up front for the whole batch: every dst copy must still hash the same.
        for r in todo:
            C.need(evidence_ok(r), f"destination evidence missing/mismatch: {r['relpath']} -> {r['dst_match']}")
        refs = C.open_refs(a.base, a.proc, a.protect_pid, a.allow_unreadable_comm)
        for i, r in enumerate(todo):
            if i and i % a.rescan == 0:
                refs = C.open_refs(a.base, a.proc, a.protect_pid, a.allow_unreadable_comm)
            rel = r["relpath"]
            try:
                p = C.safe_path(a.base, rel)
            except FileNotFoundError:
                C.need(a.allow_gone, f"source already gone: {rel} (use --allow-gone after checking the log)")
                emit("GONE", r)
                continue
            C.check_pinned(p, r)
            if p in refs:  # actively referenced -> excluded from this run, never deleted
                emit("SKIP_REFERENCED", r, "pids=" + ",".join(refs[p]))
                skipped += 1
                continue
            C.need(C.sha256_file(p) == r["sha256"], f"sha changed: {p}")
            C.check_pinned(p, r)
            refs = C.open_refs(a.base, a.proc, a.protect_pid, a.allow_unreadable_comm) if a.execute else refs
            if p in refs:
                emit("SKIP_REFERENCED", r, "pids=" + ",".join(refs[p]))
                skipped += 1
                continue
            if a.execute:
                os.unlink(p)
                emit("DELETED", r)
            else:
                emit("WOULD_DELETE", r)
            done_bytes += int(r["size"])
            n += 1
    except C.Refuse as e:
        emit("ABORT", {"relpath": "-", "size": 0, "sha256": "-"}, str(e))
        print(f"ABORT after {n} files: {e}", file=sys.stderr)
        return 3
    finally:
        log.close()
    print(f"{mode}: {n}/{len(todo)} files, {done_bytes / 2**30:.2f} GiB, skipped_referenced={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
