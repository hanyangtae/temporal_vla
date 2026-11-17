#!/usr/bin/env python3
"""Local tests for the activation archive/cleanup tools (stdlib unittest; needs `zstd` on PATH).

  python3 -I scripts/utils/activation_archive_20261117/test_activation_archive.py > /tmp/actarch_test.log 2>&1; \
      tail -3 /tmp/actarch_test.log

The real-/proc scan is fail-closed on same-uid processes with unreadable fds. On a desktop host such
processes exist (keyrings, sandboxes, ...), so tests pass the names found locally as reviewed
--allow-unreadable-comm values; the fail-closed path itself is tested on a fake /proc.
"""
import hashlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cleanup_common as C  # noqa: E402
import compare  # noqa: E402
import delete_verified  # noqa: E402


def sha(b):
    return hashlib.sha256(b).hexdigest()


def wbytes(path, b):
    with open(path, "wb") as f:
        f.write(b)


def rbytes(path):
    with open(path, "rb") as f:
        return f.read()


def rtext(path):
    with open(path) as f:
        return f.read()


def local_unreadable_comms():
    me, out = os.getuid(), set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != me:
                continue
            comm = rtext(f"/proc/{pid}/comm").strip()
            os.listdir(f"/proc/{pid}/fd")
        except PermissionError:
            out.add(comm)
        except OSError:
            pass
    return sorted(out)


ALLOW = [x for c in local_unreadable_comms() for x in ("--allow-unreadable-comm", c)]


def find_stat(root, tops):
    out = subprocess.run(["find", *tops, "-printf", "%y\\t%p\\t%s\\t%T@\\t%i\\t%n\\n"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    return "".join(sorted(out.splitlines(True), key=lambda x: x.split("\t")[1]))


def find_sha(root, tops):
    files = subprocess.run(["find", *tops, "-type", "f"], cwd=root, capture_output=True, text=True,
                           check=True).stdout.split()
    return "".join(f"{sha(rbytes(os.path.join(root, f)))}  {f}\n" for f in sorted(files))


def quiet(fn, *a, **k):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*a, **k)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.mkdtemp(prefix="actarch_")
        self.src = os.path.join(self.t, "src")
        self.dst = os.path.join(self.t, "dst")
        self.wk = os.path.join(self.t, "work")
        os.makedirs(self.wk)
        files_src = {
            "full_episodes49/cellA/same.npz": b"AAAA" * 1000,      # identical, same cell/name
            "full_episodes49/cellA/renamed.npz": b"RRRR",          # sha exists in dst under another name
            "full_episodes49/cellA/diff.npz": b"source-version",   # same name, different bytes
            "full_episodes49/cellB/only.npz": b"only-in-source",   # missing
            "latents_v2/cellA/same.npz": b"latent-bytes",          # name collision with episodes, diff bytes
            "latents_v2/cellC/PARITY.json": b"parity",             # equals dst file of SAME name in OTHER cell
        }
        files_dst = {
            "extract_v1/cellA/same.npz": b"AAAA" * 1000,
            "extract_v1/cellA/other.npz": b"RRRR",
            "extract_v1/cellA/diff.npz": b"dest-version",
            "extract_v1/cellD/PARITY.json": b"parity",
        }
        for base, fs in ((self.src, files_src), (self.dst, files_dst)):
            for rel, b in fs.items():
                p = os.path.join(base, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                wbytes(p, b)
        os.symlink(os.path.join(self.src, "full_episodes49/cellB/only.npz"),
                   os.path.join(self.src, "full_episodes49/cellB/link.npz"))
        self.files_src = files_src
        for name, root, tops in (("src", self.src, ["full_episodes49", "latents_v2"]), ("dst", self.dst, ["extract_v1"])):
            wbytes(self.w(f"{name}_stat.tsv"), find_stat(root, tops).encode())
            wbytes(self.w(f"{name}_sha.txt"), find_sha(root, tops).encode())
        self.plan = self.w("plan.tsv")
        rc = quiet(compare.main, ["--src", "50", self.w("src_stat.tsv"), self.w("src_sha.txt"),
                                  "--dst", self.w("dst_stat.tsv"), self.w("dst_sha.txt"), "--out", self.plan])
        self.assertEqual(rc, 0)
        self.plan_sha = sha(rbytes(self.plan))
        self.rows = {r["relpath"]: r for r in C.read_plan(self.plan)[0]}

    def tearDown(self):
        shutil.rmtree(self.t)

    def w(self, name):
        return os.path.join(self.wk, name)

    def delete(self, *extra, log="del.tsv"):
        args = ["--plan", self.plan, "--plan-sha256", self.plan_sha, "--host", "50", "--base", self.src,
                "--log", self.w(log), *ALLOW, *extra]
        return quiet(delete_verified.main, args)

    def log_actions(self, log="del.tsv"):
        return [ln.split("\t")[2] for ln in rtext(self.w(log)).splitlines()[1:]]

    def log_abort(self, log="del.tsv"):
        return [ln for ln in rtext(self.w(log)).splitlines() if "\tABORT\t" in ln]


class TestCompare(Fixture):
    def test_classes(self):
        k = {rel: r["klass"] for rel, r in self.rows.items()}
        self.assertEqual(k["full_episodes49/cellA/same.npz"], "IDENTICAL")
        self.assertEqual(self.rows["full_episodes49/cellA/same.npz"]["dst_match"], "extract_v1/cellA/same.npz")
        self.assertEqual(k["full_episodes49/cellA/renamed.npz"], "SHA_ONLY")
        self.assertEqual(k["full_episodes49/cellA/diff.npz"], "DIFFERENT")
        self.assertEqual(k["full_episodes49/cellB/only.npz"], "MISSING")
        self.assertEqual(k["full_episodes49/cellB/link.npz"], "SKIP_SYMLINK")
        # same <cell>/<name> as an identical file but different bytes -> never IDENTICAL
        self.assertEqual(k["latents_v2/cellA/same.npz"], "DIFFERENT")
        # byte-identical but in another cell -> not deletable as IDENTICAL (dir would restore incomplete)
        self.assertEqual(k["latents_v2/cellC/PARITY.json"], "SHA_ONLY")

    def test_missing_sha_fails_closed(self):
        lines = rtext(self.w("src_sha.txt")).splitlines(True)
        wbytes(self.w("src_sha_bad.txt"), "".join(lines[1:]).encode())
        rc = quiet(compare.main, ["--src", "50", self.w("src_stat.tsv"), self.w("src_sha_bad.txt"),
                                  "--dst", self.w("dst_stat.tsv"), self.w("dst_sha.txt"), "--out", self.w("p2.tsv")])
        self.assertEqual(rc, 1)

    def test_plan_write_once(self):
        with self.assertRaises(SystemExit):
            quiet(compare.main, ["--src", "50", self.w("src_stat.tsv"), self.w("src_sha.txt"),
                                 "--dst", self.w("dst_stat.tsv"), self.w("dst_sha.txt"), "--out", self.plan])


class TestProcScan(Fixture):
    def test_unreadable_same_uid_fails_closed(self):
        if os.getuid() == 0:
            self.skipTest("root can read any fd dir")
        fake = self.w("proc")
        os.makedirs(os.path.join(fake, "4242", "fd"))
        wbytes(os.path.join(fake, "4242", "comm"), b"mystery\n")
        os.chmod(os.path.join(fake, "4242", "fd"), 0)
        try:
            with self.assertRaises(C.Refuse):
                C.open_refs(self.src, fake)
            self.assertEqual(C.open_refs(self.src, fake, allow_comm=["mystery"]), {})
        finally:
            os.chmod(os.path.join(fake, "4242", "fd"), 0o700)

    def test_real_proc_sees_open_file(self):
        p = os.path.join(self.src, "full_episodes49/cellA/same.npz")
        with open(p, "rb"):
            refs = C.open_refs(self.src, "/proc", allow_comm=local_unreadable_comms())
        self.assertIn(p, refs)
        self.assertIn(str(os.getpid()), refs[p])


class TestDeleteIdentical(Fixture):
    target = "full_episodes49/cellA/same.npz"

    def idel(self, *extra, log="del.tsv"):
        return self.delete("--mode", "identical", "--dst-sha", self.w("dst_sha.txt"), *extra, log=log)

    def test_dry_run_deletes_nothing(self):
        self.assertEqual(self.idel(), 0, self.log_abort())
        self.assertEqual(self.log_actions(), ["WOULD_DELETE"])
        self.assertTrue(os.path.exists(os.path.join(self.src, self.target)))

    def test_execute_deletes_only_identical(self):
        self.assertEqual(self.idel("--execute"), 0, self.log_abort())
        self.assertFalse(os.path.exists(os.path.join(self.src, self.target)))
        for rel in self.files_src:
            if rel != self.target:
                self.assertTrue(os.path.exists(os.path.join(self.src, rel)), rel)

    def test_wrong_plan_sha_refuses(self):
        with self.assertRaises(C.Refuse):
            quiet(delete_verified.main, ["--plan", self.plan, "--plan-sha256", "0" * 64, "--host", "50",
                                         "--base", self.src, "--log", self.w("x.tsv"), "--mode", "identical",
                                         "--dst-sha", self.w("dst_sha.txt"), "--execute"])

    def test_destination_changed_aborts(self):
        wbytes(self.w("dst_sha2.txt"), rtext(self.w("dst_sha.txt")).replace(sha(b"AAAA" * 1000), "f" * 64).encode())
        rc = self.delete("--mode", "identical", "--dst-sha", self.w("dst_sha2.txt"), "--execute")
        self.assertEqual(rc, 3)
        self.assertIn("destination evidence", self.log_abort()[0])
        self.assertTrue(os.path.exists(os.path.join(self.src, self.target)))

    def test_mtime_change_aborts(self):
        p = os.path.join(self.src, self.target)
        st = os.stat(p)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
        self.assertEqual(self.idel("--execute"), 3)
        self.assertIn("mtime changed", self.log_abort()[0])
        self.assertTrue(os.path.exists(p))

    def test_content_change_same_meta_aborts(self):
        p = os.path.join(self.src, self.target)
        st = os.stat(p)
        with open(p, "r+b") as f:
            f.write(b"B")
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
        self.assertEqual(self.idel("--execute"), 3)
        self.assertIn("sha changed", self.log_abort()[0])
        self.assertTrue(os.path.exists(p))

    def test_symlinked_parent_aborts(self):
        cell = os.path.join(self.src, "full_episodes49/cellA")
        os.rename(cell, cell + "_real")
        os.symlink(cell + "_real", cell)
        self.assertEqual(self.idel("--execute"), 3)
        self.assertIn("symlink in path", self.log_abort()[0])
        self.assertTrue(os.path.exists(os.path.join(cell + "_real", "same.npz")))

    def test_open_file_is_excluded_not_deleted(self):
        p = os.path.join(self.src, self.target)
        with open(p, "rb"):
            self.assertEqual(self.idel("--execute"), 0, self.log_abort())
        self.assertEqual(self.log_actions(), ["SKIP_REFERENCED"])
        self.assertTrue(os.path.exists(p))

    def test_dead_protected_pid_aborts(self):
        fake_proc = self.w("proc")
        os.makedirs(fake_proc)
        self.assertEqual(self.idel("--execute", "--protect-pid", "999999", "--proc", fake_proc), 3)
        self.assertIn("protected pid", self.log_abort()[0])
        self.assertTrue(os.path.exists(os.path.join(self.src, self.target)))

    def test_gone_requires_flag(self):
        os.unlink(os.path.join(self.src, self.target))
        self.assertEqual(self.idel("--execute"), 3)
        self.assertEqual(self.idel("--execute", "--allow-gone", log="del2.tsv"), 0, self.log_abort("del2.tsv"))
        self.assertEqual(self.log_actions("del2.tsv"), ["GONE"])

    def test_log_write_once(self):
        self.idel()
        with self.assertRaises(SystemExit):
            self.idel()


@unittest.skipUnless(shutil.which("zstd"), "zstd missing")
class TestPackVerifyDelete(Fixture):
    def pack_cmd(self, root, man):
        return [sys.executable, "-I", os.path.join(HERE, "pack_transfer.py"), "--plan", self.plan,
                "--plan-sha256", self.plan_sha, "--host", "50", "--root", root, "--base", self.src,
                "--manifest", man, *ALLOW]

    def pack(self, root, name):
        man, arc = self.w(f"{name}.manifest.tsv"), self.w(f"{name}.tar.zst")
        with open(self.w(f"{name}.pack.err"), "wb") as err:
            p1 = subprocess.Popen(self.pack_cmd(root, man), stdout=subprocess.PIPE, stderr=err)
            assert p1.stdout is not None
            p2 = subprocess.run(["zstd", "-q", "-T2", "-10", "--check", "-o", arc], stdin=p1.stdout)
            p1.stdout.close()
            rc = p1.wait()
        self.assertEqual(rc, 0, rtext(self.w(f"{name}.pack.err")))
        self.assertEqual(p2.returncode, 0)
        return man, arc

    def verify(self, man, arc, out, extract=None):
        cmd = [sys.executable, "-I", os.path.join(HERE, "verify_archive.py"), "--manifest", man, "--out", out]
        if extract:
            cmd += ["--extract-to", extract]
        d = subprocess.Popen(["zstd", "-dcq", arc], stdout=subprocess.PIPE)
        assert d.stdout is not None
        r = subprocess.run(cmd, stdin=d.stdout, stderr=subprocess.DEVNULL)
        d.stdout.close()
        d.wait()
        return r.returncode

    def packed(self, man):
        return sorted(ln.split("\t")[1] for ln in rtext(man).splitlines() if ln.startswith("PACKED"))

    def test_roundtrip_then_transferred_delete(self):
        man, arc = self.pack("full_episodes49", "fe")
        packed = self.packed(man)
        self.assertEqual(packed, ["full_episodes49/cellA/diff.npz", "full_episodes49/cellA/renamed.npz",
                                  "full_episodes49/cellB/only.npz"])
        ext = os.path.join(self.t, "restore")
        self.assertEqual(self.verify(man, arc, self.w("fe.verified.tsv"), extract=ext), 0)
        for rel in packed:
            self.assertEqual(rbytes(os.path.join(ext, rel)), self.files_src[rel])
        self.assertEqual(self.delete("--mode", "transferred", "--verified", self.w("fe.verified.tsv"), "--execute"), 0,
                         self.log_abort())
        for rel in packed:
            self.assertFalse(os.path.exists(os.path.join(self.src, rel)))
        # rows not in this archive's verified list (other root, IDENTICAL) untouched
        self.assertTrue(os.path.exists(os.path.join(self.src, "latents_v2/cellA/same.npz")))
        self.assertTrue(os.path.exists(os.path.join(self.src, "latents_v2/cellC/PARITY.json")))
        self.assertTrue(os.path.exists(os.path.join(self.src, "full_episodes49/cellA/same.npz")))

    def test_extract_target_must_not_exist(self):
        man, arc = self.pack("full_episodes49", "fe")
        ext = os.path.join(self.t, "exists")
        os.makedirs(ext)
        self.assertNotEqual(self.verify(man, arc, self.w("v.tsv"), extract=ext), 0)

    def test_tampered_manifest_fails(self):
        man, arc = self.pack("full_episodes49", "fe")
        wbytes(self.w("bad.manifest.tsv"), rtext(man).replace(sha(b"only-in-source"), "e" * 64).encode())
        self.assertEqual(self.verify(self.w("bad.manifest.tsv"), arc, self.w("bad.verified.tsv")), 4)
        with self.assertRaises(SystemExit):  # failed verification is not accepted as deletion evidence
            self.delete("--mode", "transferred", "--verified", self.w("bad.verified.tsv"), "--execute")

    def test_source_changed_before_pack_aborts(self):
        p = os.path.join(self.src, "full_episodes49/cellB/only.npz")
        st = os.stat(p)
        os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5))
        man = self.w("x.manifest.tsv")
        with open(self.w("x.tar"), "wb") as out:
            r = subprocess.run(self.pack_cmd("full_episodes49", man), stdout=out, stderr=subprocess.DEVNULL)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("# ABORT", rtext(man))
        self.assertIn("mtime changed", rtext(man))


if __name__ == "__main__":
    unittest.main()
