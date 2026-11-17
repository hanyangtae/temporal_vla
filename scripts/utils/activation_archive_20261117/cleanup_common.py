"""Shared fail-closed checks for the fold27 activation cleanup (stdlib only, py3.8+).

Used by delete_verified.py (source deletion) and pack_transfer.py (lossless packing). Nothing here
mutates files.
"""
import csv
import hashlib
import os
import stat

BUF = 8 << 20
# Same-uid processes whose /proc/<pid>/fd is unreadable by design (non-dumpable). Anything else
# unreadable -> fail-closed, because we cannot prove it does not hold a target open.
BENIGN_UNREADABLE_COMM = {"sshd", "(sd-pam)"}


class Refuse(Exception):
    """Fail-closed condition; caller must stop (no partial-progress guessing)."""


def need(cond, msg):
    if not cond:
        raise Refuse(msg)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(BUF)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def mtime_ns_from_find(s):
    """find -printf %T@ -> 'SEC.FRACTION' (10 fractional digits on GNU find). Exact to the ns."""
    sec, _, frac = s.partition(".")
    return int(sec) * 1_000_000_000 + int((frac + "000000000")[:9])


def read_plan(path, expect_sha=None):
    raw = open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    if expect_sha is not None:
        need(digest == expect_sha, f"plan sha256 {digest} != expected {expect_sha}")
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return rows, digest


def safe_path(base, rel):
    """Join base/rel refusing absolute paths, '..', and symlinks in ANY component (base included)."""
    need(not os.path.isabs(rel) and ".." not in rel.split("/") and "\n" not in rel, f"unsafe relpath {rel!r}")
    need(os.path.isabs(base), f"base must be absolute: {base}")
    cur = "/"
    for part in [p for p in base.split("/") if p] + rel.split("/"):
        cur = os.path.join(cur, part)
        st = os.lstat(cur)
        need(not stat.S_ISLNK(st.st_mode), f"symlink in path: {cur}")
    return cur


def check_pinned(path, row):
    """lstat must match the plan pin exactly: regular, single link, size, mtime_ns, inode."""
    st = os.lstat(path)
    need(stat.S_ISREG(st.st_mode), f"not a regular file: {path}")
    need(st.st_nlink == 1, f"nlink {st.st_nlink} != 1: {path}")
    need(st.st_size == int(row["size"]), f"size changed: {path}")
    need(st.st_ino == int(row["inode"]), f"inode changed: {path}")
    need(st.st_mtime_ns == mtime_ns_from_find(row["mtime"]), f"mtime changed: {path}")
    return st


def open_refs(prefix, proc="/proc", protect=(), allow_comm=()):
    """Return {path: [pid,...]} for every fd/cwd/mmap under `prefix` held by any readable process.

    Raises Refuse if a same-uid process has unreadable fds and its comm is neither in
    BENIGN_UNREADABLE_COMM nor in the explicitly reviewed `allow_comm`, or if any protected pid is gone
    (state changed since review) — fail-closed instead of `fuser`.
    """
    prefix = prefix.rstrip("/") + "/"
    me = os.getuid()
    benign = BENIGN_UNREADABLE_COMM | set(allow_comm)
    refs = {}
    for pid in os.listdir(proc):
        if not pid.isdigit():
            continue
        base = os.path.join(proc, pid)
        try:
            uid = os.stat(base).st_uid
            comm = open(os.path.join(base, "comm")).read().strip()
        except OSError:
            continue
        try:
            fds = [os.path.join(base, "fd", x) for x in os.listdir(os.path.join(base, "fd"))]
        except PermissionError:
            if uid == me and comm not in benign:
                raise Refuse(f"cannot read fds of same-uid pid {pid} ({comm})")
            fds = []
        except OSError:
            continue
        links = fds + [os.path.join(base, "cwd")]
        for ln in links:
            try:
                t = os.readlink(ln)
            except OSError:
                continue
            if t.startswith(prefix):
                refs.setdefault(t, []).append(pid)
        try:
            with open(os.path.join(base, "maps")) as f:
                for line in f:
                    parts = line.split(None, 5)
                    if len(parts) == 6 and parts[5].strip().startswith(prefix):
                        refs.setdefault(parts[5].strip(), []).append(pid)
        except OSError:
            pass
    for p in protect:
        need(os.path.isdir(os.path.join(proc, str(p))), f"protected pid {p} not alive (state changed; re-review)")
        for t, pids in refs.items():
            need(str(p) not in pids, f"protected pid {p} holds {t}")
    return refs
