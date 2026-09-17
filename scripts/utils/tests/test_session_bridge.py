"""Transport/recipient correctness tests; no model calls or existing-session writes."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("session_bridge", Path(__file__).parents[1] / "session_bridge.py")
b = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(b)


class BridgeTests(unittest.TestCase):
    def test_ambiguous_names_require_identity(self):
        rows = [{"provider": p, "id": p + "-12345678", "name": "worker"} for p in ["claude", "codex"]]
        with patch.object(b, "sessions", return_value=(rows, [])):
            with self.assertRaisesRegex(b.BridgeError, "Ambiguous"):
                b.resolve("worker")
            self.assertEqual(b.resolve("claude:worker")["provider"], "claude")

    def test_exact_id_wins_over_same_name(self):
        rows = [{"provider": "claude", "id": "12345678", "name": "worker"},
                {"provider": "codex", "id": "abcdef12", "name": "12345678"}]
        with patch.object(b, "sessions", return_value=(rows, [])):
            self.assertEqual(b.resolve("12345678")["provider"], "claude")

    def test_rejects_empty_and_oversized_utf8(self):
        for message in [" ", "가" * 11000, None]:
            with self.assertRaises(b.BridgeError):
                b.send("any", message)

    def test_private_key_rejects_symlink_and_public_mode(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "key"
            path.write_text('{}')
            path.chmod(0o644)
            with self.assertRaises(b.BridgeError):
                b.private_json(path)
            path.chmod(0o600)
            self.assertEqual(b.private_json(path), {})
            link = Path(root) / "link"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                b.private_json(link)

    def test_real_peer_socket_receives_auth_and_literal_message(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / "sessions").mkdir()
            path = root / "peer.sock"
            server = socket.socket(socket.AF_UNIX)
            self.addCleanup(server.close)
            server.bind(str(path))
            path.chmod(0o600)
            server.listen(1)
            received = []

            def receive():
                conn, _ = server.accept()
                with conn:
                    data = b""
                    while chunk := conn.recv(4096):
                        data += chunk
                    received.extend(json.loads(line) for line in data.splitlines())

            thread = threading.Thread(target=receive, daemon=True)
            thread.start()
            pid = os.getpid()
            meta = {"pid": pid, "procStart": b.process_start(pid), "pidDomain": "test",
                    "messagingSocketPath": str(path)}
            key = root / "sessions" / f"{pid}.{hashlib.sha256(str(path).encode()).hexdigest()}.key"
            key.write_text(json.dumps({"peerToken": "a" * 32, "procStart": meta["procStart"], "pidDomain": "test"}))
            key.chmod(0o600)
            row = {"_registry": meta, "id": "test-session", "sendable": True}
            message = '한글\n$(touch /tmp/do-not-create) `echo literal`'
            with patch.object(b, "CLAUDE", root):
                result = b.send_claude(row, message, "test-message")
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(received[0], {"type": "auth", "token": "a" * 32})
            self.assertEqual(received[1]["message"]["content"], message)
            self.assertEqual(received[1]["session_id"], "test-session")
            self.assertEqual(result["status"], "sent_unconfirmed")

    def test_stale_process_does_not_send(self):
        row = {"sendable": True, "_registry": {"pid": os.getpid(), "procStart": "stale", "messagingSocketPath": "/missing"}}
        with self.assertRaisesRegex(b.BridgeError, "process changed"):
            b.send_claude(row, "test", "id")

    def test_codex_not_loaded_cannot_accidentally_resume(self):
        row = {"provider": "codex", "id": "test", "sendable": False}
        with patch.object(b, "resolve", return_value=row), patch.object(b.subprocess, "run") as run:
            with self.assertRaisesRegex(b.BridgeError, "not loaded"):
                b.send("test", "work")
            run.assert_not_called()

    def test_codex_queue_uses_argv_and_no_permission_flags(self):
        row = {"provider": "codex", "id": "test", "sendable": True}
        done = b.subprocess.CompletedProcess([], 0, "Queued", "")
        with patch.object(b, "resolve", return_value=row), patch.object(b.subprocess, "run", return_value=done) as run:
            b.send("test", "literal `date`\n한글")
            argv = run.call_args.args[0]
            self.assertEqual(argv[:2], ["codex", "queue"])
            self.assertTrue(argv[-1].endswith("literal `date`\n한글"))
            self.assertFalse(any("dangerously" in arg or "approval" in arg for arg in argv))

    def test_mcp_frames_and_tool_error(self):
        requests = [{"id": 1, "method": "initialize"}, {"method": "notifications/initialized"},
                    {"id": 2, "method": "tools/list"},
                    {"id": 3, "method": "tools/call", "params": {"name": "unknown"}}]
        out = io.StringIO()
        with patch.object(b.sys, "stdin", io.StringIO("\n".join(map(json.dumps, requests)))), contextlib.redirect_stdout(out):
            b.mcp()
        results = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([r["id"] for r in results], [1, 2, 3])
        self.assertEqual(len(results[1]["result"]["tools"]), 3)
        self.assertTrue(results[2]["result"]["isError"])

    def test_managed_start_uses_remote_control(self):
        done = b.subprocess.CompletedProcess([], 0, '{}', '')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "managed.sock"
            with patch.object(b, "CODEX_SOCKET", path), patch.object(b, "MANAGED_CODEX_SOCKET", path), \
                 patch.object(b, "CodexRPC", side_effect=[ConnectionRefusedError, unittest.mock.MagicMock()]), \
                 patch.object(b.subprocess, "run", return_value=done) as run:
                self.assertTrue(b._start_codex_server()["managed"])
                self.assertEqual(run.call_args.args[0], ["codex", "remote-control", "start", "--json"])

    def test_missing_custom_socket_never_starts_server(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(b, "CODEX_SOCKET", Path(root) / "custom.sock"), \
                 patch.object(b, "CodexRPC", side_effect=ConnectionRefusedError), \
                 patch.object(b.subprocess, "run") as run:
                with self.assertRaisesRegex(b.BridgeError, "Custom Codex socket"):
                    b._start_codex_server()
                run.assert_not_called()

    def test_start_does_not_replace_a_live_unresponsive_server(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "codex.sock"
            with socket.socket(socket.AF_UNIX) as server:
                server.bind(str(path))
                server.listen(1)
                with patch.object(b, "CODEX_SOCKET", path), patch.object(b, "CodexRPC", side_effect=TimeoutError), patch.object(b.subprocess, "Popen") as spawn:
                    with self.assertRaisesRegex(b.BridgeError, "owns this socket"):
                        b._start_codex_server()
                    spawn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
