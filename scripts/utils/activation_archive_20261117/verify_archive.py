#!/usr/bin/env python3
"""Restore-verify a transferred archive against its pack manifest (lossless proof).

  zstd -dc --check X.tar.zst | python3 verify_archive.py --manifest X.manifest.tsv --out X.verified.tsv \
      [--extract-to /new/dir]

Reads the tar stream from stdin, hashes every member's decompressed bytes, and requires the member set
to equal the manifest's PACKED rows exactly (same relpaths, sizes, sha256). Members must be regular
files with safe relative names (no abs path, '..', links, devices). With --extract-to (must NOT
exist), files are also written there and re-hashed from disk after close. Writes a write-once
verified TSV whose rows are what delete_verified.py --mode transferred accepts as evidence.
Exit 0 only on a complete, exact match.
"""
import argparse
import hashlib
import os
import sys
import tarfile

BUF = 8 << 20


def read_manifest(path):
    rows = {}
    with open(path) as f:
        for ln in f:
            if ln.startswith("#") or ln.startswith("status\t"):
                if ln.startswith("# ABORT"):
                    raise SystemExit(f"manifest records an aborted pack: {ln.strip()}")
                continue
            st, rel, size, sha = ln.rstrip("\n").split("\t")
            if st == "PACKED":
                rows[rel] = (int(size), sha)
    return rows


def safe_name(n):
    return n and not os.path.isabs(n) and ".." not in n.split("/") and "\n" not in n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extract-to")
    a = ap.parse_args(argv)
    want = read_manifest(a.manifest)
    if a.extract_to:
        os.makedirs(a.extract_to)  # raises if it exists: never overwrite
    out = open(a.out, "x")
    out.write("relpath\tsize\tsha256\n")
    seen, bad = {}, []
    with tarfile.open(fileobj=sys.stdin.buffer, mode="r|") as tar:
        for m in tar:
            if not m.isreg() or not safe_name(m.name):
                bad.append(f"unsafe/non-regular member {m.name!r}")
                continue
            if m.name in seen:
                bad.append(f"duplicate member {m.name}")
                continue
            f = tar.extractfile(m)
            if f is None:
                bad.append(f"unreadable member {m.name}")
                continue
            h = hashlib.sha256()
            dst, tgt = None, None
            if a.extract_to:
                tgt = os.path.join(a.extract_to, m.name)
                os.makedirs(os.path.dirname(tgt), exist_ok=True)
                dst = open(tgt, "xb")
            n = 0
            while True:
                b = f.read(BUF)
                if not b:
                    break
                h.update(b)
                n += len(b)
                if dst:
                    dst.write(b)
            if dst is not None and tgt is not None:
                dst.close()
                h2 = hashlib.sha256()
                with open(tgt, "rb") as rf:
                    for b in iter(lambda: rf.read(BUF), b""):
                        h2.update(b)
                h2 = h2.hexdigest()
                if h2 != h.hexdigest():
                    bad.append(f"on-disk sha differs after extract: {m.name}")
            seen[m.name] = (n, h.hexdigest())
    for rel, (size, sha) in want.items():
        if rel not in seen:
            bad.append(f"missing member {rel}")
        elif seen[rel] != (size, sha):
            bad.append(f"size/sha mismatch {rel}: {seen[rel]} != {(size, sha)}")
    for rel in seen:
        if rel not in want:
            bad.append(f"unexpected member {rel}")
    if bad:
        out.write(f"# FAILED {len(bad)}: {bad[0]}\n")
        out.close()
        print(f"FAILED ({len(bad)}): " + "; ".join(bad[:5]), file=sys.stderr)
        return 4
    for rel in sorted(seen):
        out.write(f"{rel}\t{seen[rel][0]}\t{seen[rel][1]}\n")
    out.close()
    print(f"VERIFIED {len(seen)} members", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
