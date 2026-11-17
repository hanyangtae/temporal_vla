#!/usr/bin/env python3
"""Pack plan rows that have NO identical Seungjun copy (klass MISSING/DIFFERENT/SHA_ONLY) into an uncompressed
tar stream on stdout, for lossless zstd compression + transfer to a NEW destination path.

  python3 pack_transfer.py --plan plan.tsv --plan-sha256 <sha> --host 50 --root full_episodes \
      --base /tmp/temporal-vla-fold27-production-20261102/outputs/analysis/groot_fold27_20261028 \
      --manifest srv50_full_episodes.manifest.tsv \
    | zstd -T4 -10 --check -o srv50_full_episodes.tar.zst      # -o refuses to overwrite without -f

Per member: safe path (no symlink component), pinned size/mtime_ns/inode/nlink, not open by any
process (open -> EXCLUDED, recorded, left on disk, never deleted later), sha256 computed on the
exact bytes written into the tar and required == plan sha. Any other failure -> abort (nonzero exit,
truncated stream; zstd output must then be discarded). Tar members keep relpath under --base, mode
and mtime; uid/gid/uname zeroed (contents only). Manifest (write-once) lists every packed member:
relpath size sha256 — verify_archive.py checks the restored bytes against it.
"""
import argparse
import hashlib
import os
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cleanup_common as C  # noqa: E402

TRANSFER_KLASSES = ("MISSING", "DIFFERENT", "SHA_ONLY")


class HashingReader:
    def __init__(self, f):
        self.f, self.h, self.n = f, hashlib.sha256(), 0

    def read(self, size=-1):
        b = self.f.read(size)
        self.h.update(b)
        self.n += len(b)
        return b


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--plan-sha256", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--root", required=True, help="top-level dir in plan relpath (e.g. full_episodes)")
    ap.add_argument("--base", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--protect-pid", action="append", default=[])
    ap.add_argument("--allow-unreadable-comm", action="append", default=[],
                    help="reviewed same-uid process names whose fds may be unreadable (sshd,(sd-pam) built in)")
    ap.add_argument("--proc", default="/proc", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if sys.stdout.isatty():
        raise SystemExit("refusing to write tar to a terminal; pipe into zstd")
    rows, plan_sha = C.read_plan(a.plan, a.plan_sha256)
    todo = [r for r in rows if r["host"] == a.host and r["klass"] in TRANSFER_KLASSES
            and r["relpath"].split("/", 1)[0] == a.root]
    if not todo:
        raise SystemExit("no rows to pack")
    man = open(a.manifest, "x")
    man.write(f"# plan_sha256={plan_sha} host={a.host} root={a.root}\nstatus\trelpath\tsize\tsha256\n")
    n = skipped = 0
    try:
        refs = C.open_refs(a.base, a.proc, a.protect_pid, a.allow_unreadable_comm)
        with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", format=tarfile.PAX_FORMAT) as tar:
            for r in todo:
                p = C.safe_path(a.base, r["relpath"])
                st = C.check_pinned(p, r)
                if p in refs:
                    man.write(f"EXCLUDED_REFERENCED\t{r['relpath']}\t{r['size']}\t{r['sha256']}\n")
                    skipped += 1
                    continue
                ti = tarfile.TarInfo(r["relpath"])
                ti.size, ti.mode, ti.mtime = st.st_size, st.st_mode & 0o7777, st.st_mtime_ns / 1e9
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                with open(p, "rb") as f:
                    hr = HashingReader(f)
                    tar.addfile(ti, hr)
                C.need(hr.n == st.st_size, f"short read {p}")
                C.need(hr.h.hexdigest() == r["sha256"], f"sha changed while packing: {p}")
                C.check_pinned(p, r)
                man.write(f"PACKED\t{r['relpath']}\t{r['size']}\t{r['sha256']}\n")
                man.flush()
                n += 1
    except C.Refuse as e:
        man.write(f"# ABORT {e}\n")
        print(f"ABORT after {n} members: {e}", file=sys.stderr)
        return 3
    finally:
        man.close()
    print(f"packed {n}/{len(todo)} members, excluded_referenced={skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
