"""TCP regressions for restarting completed evaluation servers on one port."""

import errno
import socket
import subprocess
import unittest
from unittest.mock import patch

try:
    from . import eval as module
except ImportError:
    import eval as module


class PortAvailabilityTests(unittest.TestCase):
    def listener(self):
        server = socket.socket()
        self.addCleanup(server.close)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        return server

    def closed_server_port(self):
        """Leave a genuine server-side TIME_WAIT socket, without sleeps."""
        server = self.listener()
        port = server.getsockname()[1]
        client = socket.socket()
        self.addCleanup(client.close)
        client.settimeout(2)
        client.connect(("127.0.0.1", port))
        accepted, _ = server.accept()
        self.addCleanup(accepted.close)
        accepted.settimeout(2)
        # The server actively closes first, so TIME_WAIT belongs to its port.
        accepted.shutdown(socket.SHUT_WR)
        self.assertEqual(client.recv(1), b"")
        client.close()
        self.assertEqual(accepted.recv(1), b"")
        accepted.close()
        server.close()
        # Prove the pre-fix probe actually fails; an unused ephemeral port
        # would conceal the regression that interrupted the live queue.
        with socket.socket() as old_probe:
            with self.assertRaises(OSError) as raised:
                old_probe.bind(("0.0.0.0", port))
            self.assertEqual(raised.exception.errno, errno.EADDRINUSE)
        return port

    def idle_snapshots(self):
        return patch.object(module.subprocess, "run", side_effect=[
            subprocess.CompletedProcess([], 0, stdout=""),
            subprocess.CompletedProcess([], 0, stdout="PID COMMAND\n"),
        ])

    def test_closed_server_time_wait_can_be_reused(self):
        port = self.closed_server_port()
        module.check_port_available(port)
        # Probing must leave no listener or reservation behind.
        with socket.socket() as replacement:
            replacement.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            replacement.bind(("0.0.0.0", port))
            replacement.listen(1)

    def test_live_listener_is_rejected_even_when_it_sets_reuseaddr(self):
        server = self.listener()
        with self.assertRaises(OSError) as raised:
            module.check_port_available(server.getsockname()[1])
        self.assertEqual(raised.exception.errno, errno.EADDRINUSE)

    def test_idle_gpu_check_accepts_closed_server_time_wait(self):
        port = self.closed_server_port()
        with self.idle_snapshots() as snapshots:
            module.check_idle(6, port)
        self.assertEqual(snapshots.call_count, 2)

    def test_idle_gpu_does_not_allow_a_live_listener(self):
        server = self.listener()
        with self.idle_snapshots():
            with self.assertRaises(OSError) as raised:
                module.check_idle(6, server.getsockname()[1])
        self.assertEqual(raised.exception.errno, errno.EADDRINUSE)


if __name__ == "__main__":
    unittest.main()
