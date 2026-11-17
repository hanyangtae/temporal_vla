#!/usr/bin/env python3
"""Classify source files (srv48/srv50) against Seungjun's existing copies by SHA256 (local, read-only).

Inputs are the manifest pairs produced on each host by read-only shell:
  stat : find <roots> -printf "%y\\t%p\\t%s\\t%T@\\t%i\\t%n\\n"     (type, relpath, size, mtime, inode, nlink)
  sha  : find <roots> -type f -print0 | sort -z | xargs -0 sha256sum  ("<sha>  <relpath>")

Classes (per source regular file):
  IDENTICAL   a Seungjun file with same sha256 AND size AND same <cell>/<basename> exists -> source deletable
  SHA_ONLY    same sha+size exists in Seungjun but under another <cell>/<basename> (e.g. one cell's
              PARITY_SMOKE_V2.json equal to another cell's) -> transfer, so each dir restores complete
  DIFFERENT   no sha match, but a Seungjun file with the same <cell>/<basename> exists (typically a
              different extraction version with the same sig name, not corruption)  -> transfer
  MISSING     no sha match, no same-name file                                       -> transfer
Non-regular source entries: symlinks -> SKIP_SYMLINK (never touched); other -> SKIP_OTHER.
A regular file in stat without a sha line (or vice versa) -> ERROR (fail-closed: whole plan refuses).

Usage:
  compare.py --src 48 manifests/src48_stat.tsv manifests/src48_sha256.txt \
             --src 50 manifests/src50_stat.tsv manifests/src50_sha256.txt \
             --dst manifests/sj_stat_main.tsv manifests/sj_sha_main.txt --out plan.tsv
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import sys

COLS = ["host", "relpath", "size", "mtime", "inode", "nlink", "sha256", "klass", "dst_match", "note"]


def read_stat(path):
    rows = {}
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            t, rel, size, mtime, ino, nlink = ln.split("\t")
            rows[rel] = dict(type=t, size=int(size), mtime=mtime, inode=ino, nlink=int(nlink))
    return rows


def read_sha(path):
    out = {}
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if not ln:
                continue
            digest, rel = ln.split("  ", 1)
            if len(digest) != 64 or rel in out:
                raise SystemExit(f"bad/duplicate sha line in {path}: {ln[:120]}")
            out[rel] = digest
    return out


def key_name(rel):
    return os.path.join(os.path.basename(os.path.dirname(rel)), os.path.basename(rel))


def classify(srcs, dst_stat, dst_sha):
    errors = []
    by_sha = collections.defaultdict(list)
    by_name = collections.defaultdict(list)
    for rel, st in dst_stat.items():
        if st["type"] != "f":
            continue
        if rel not in dst_sha:
            errors.append(f"dst stat file without sha: {rel}")
            continue
        by_sha[(dst_sha[rel], st["size"])].append(rel)
        by_name[key_name(rel)].append(rel)
    for rel in dst_sha:
        if rel not in dst_stat:
            errors.append(f"dst sha without stat: {rel}")
    rows = []
    for host, stat, sha in srcs:
        for rel in sha:
            if rel not in stat:
                errors.append(f"{host}: sha without stat: {rel}")
        for rel in sorted(stat):
            st = stat[rel]
            if st["type"] == "d":
                continue
            base = dict(host=host, relpath=rel, size=st["size"], mtime=st["mtime"], inode=st["inode"],
                        nlink=st["nlink"], sha256="-", dst_match="-", note="")
            if st["type"] == "l":
                rows.append({**base, "klass": "SKIP_SYMLINK"})
                continue
            if st["type"] != "f":
                rows.append({**base, "klass": "SKIP_OTHER", "note": f"type={st['type']}"})
                continue
            if rel not in sha:
                errors.append(f"{host}: stat file without sha: {rel}")
                rows.append({**base, "klass": "ERROR", "note": "no sha"})
                continue
            d = sha[rel]
            base["sha256"] = d
            notes = []
            if st["nlink"] != 1:
                notes.append(f"nlink={st['nlink']}")
            if os.path.basename(rel).startswith("."):
                notes.append("hidden/partial")
            hits = by_sha.get((d, st["size"]), [])
            same_key = [h for h in hits if key_name(h) == key_name(rel)]
            if same_key:
                rows.append({**base, "klass": "IDENTICAL", "dst_match": same_key[0], "note": ";".join(notes)})
            elif hits:
                rows.append({**base, "klass": "SHA_ONLY", "dst_match": hits[0], "note": ";".join(notes)})
            elif by_name.get(key_name(rel)):
                rows.append({**base, "klass": "DIFFERENT", "dst_match": by_name[key_name(rel)][0], "note": ";".join(notes)})
            else:
                rows.append({**base, "klass": "MISSING", "note": ";".join(notes)})
    return rows, errors


def summarize(rows):
    agg = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        top = r["relpath"].split("/", 1)[0]
        for k in ((r["host"], top, r["klass"]), (r["host"], "*", r["klass"])):
            agg[k][0] += 1
            agg[k][1] += int(r["size"])
    return [dict(host=h, root=t, klass=k, files=n, bytes=b) for (h, t, k), (n, b) in sorted(agg.items())]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs=3, action="append", metavar=("HOST", "STAT", "SHA"), required=True)
    ap.add_argument("--dst", nargs=2, metavar=("STAT", "SHA"), required=True)
    ap.add_argument("--out", required=True, help="plan TSV (must not exist)")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} exists; plans are write-once")
    srcs = [(h, read_stat(s), read_sha(x)) for h, s, x in a.src]
    rows, errors = classify(srcs, read_stat(a.dst[0]), read_sha(a.dst[1]))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    summ = summarize(rows)
    inputs = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for trio in a.src for p in trio[1:]}
    inputs.update({p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in a.dst})
    meta = {"plan": os.path.basename(a.out), "plan_sha256": hashlib.sha256(open(a.out, "rb").read()).hexdigest(),
            "inputs_sha256": inputs, "errors": errors, "summary": summ}
    with open(a.out + ".summary.json", "w") as f:
        json.dump(meta, f, indent=1)
    for s in summ:
        if s["root"] == "*":
            print(f"{s['host']}\t{s['klass']:<13}\t{s['files']:>6} files\t{s['bytes'] / 2**30:8.2f} GiB")
    if errors:
        print(f"ERRORS: {len(errors)} (first: {errors[0]})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
