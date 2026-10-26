"""A managed-runtime relaunch must not consume the one-shot worker token."""
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

import pytest

from agent.isolated_protocol import receive


@pytest.mark.platforms("posix")
def test_runtime_relaunch_precedes_private_handshake(tmp_path):
    marker = tmp_path / "runtime-ready"
    (tmp_path / "hermes_bootstrap.py").write_text(
        "import os, sys\nfrom pathlib import Path\n"
        f"marker = Path({str(marker)!r})\n"
        "if not marker.exists():\n"
        "    marker.write_text(str(os.getpid()))\n"
        "    os.execv(sys.executable, [sys.executable, '-m', 'agent.isolated_worker', sys.argv[1]])\n"
    )
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(tmp_path) + os.pathsep + str(root))
    token = "a" * 64
    with tempfile.TemporaryDirectory(prefix="hboot-") as directory:
        endpoint = str(Path(directory) / "control")
        with socket.socket(socket.AF_UNIX) as listener:
            listener.bind(endpoint)
            listener.listen(1)
            listener.settimeout(10)
            child = subprocess.Popen(
                [sys.executable, "-m", "agent.isolated_worker", endpoint],
                cwd=tmp_path, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                child.stdin.write(token + "\n")
                child.stdin.flush()
                with listener.accept()[0] as peer:
                    peer.settimeout(10)
                    hello = receive(peer)
                    assert hello["token"] == token
                    assert marker.exists(), "worker authenticated before managed-runtime activation"
                    assert int(marker.read_text()) == child.pid
            finally:
                child.terminate()
                child.communicate(timeout=10)
