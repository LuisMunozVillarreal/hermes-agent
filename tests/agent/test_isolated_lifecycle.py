"""Cancellation and authority loss across the real private-process boundary."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import socket
import subprocess
import sys
import threading

import pytest

from agent.isolated_agent import IsolatedAgent
from agent.isolated_protocol import receive

pytestmark = pytest.mark.platforms("linux")
ROOT = Path(__file__).resolve().parents[2]


def test_cancellation_interrupts_bootstrap_wait(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    launched = threading.Event()
    def launch(args, **options):
        # A real child held before authentication, not a fake process handle.
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], **options)
        launched.set()
        return process
    child = IsolatedAgent(kind='delegation', launcher=launch)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(child.run_conversation, 'must not run')
        try:
            assert launched.wait(3)
            child.interrupt('bootstrap cancelled')
            assert future.result(timeout=3)['interrupted']
            assert child.process.poll() is not None
        finally:
            child.interrupt()
            future.result(timeout=8)
            child.close()


def test_invalid_bootstrap_nonce_never_receives_construction(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    marker = tmp_path / 'received-secret'
    code = (
        'import socket,sys; from pathlib import Path; from agent.isolated_protocol import send; '
        'sys.stdin.readline(); s=socket.socket(socket.AF_UNIX); s.connect(sys.argv[1]); '
        "send(s, {'v':1,'op':'hello','token':'0'*64}); "
        "data=s.recv(65536); Path(sys.argv[2]).write_bytes(data)"
    )
    def launch(args, **options):
        return subprocess.Popen([sys.executable, '-c', code, args[-1], str(marker)], **options)
    child = IsolatedAgent(kind='delegation', launcher=launch, api_key='private-runtime-key')
    with pytest.raises(PermissionError, match='authentication'):
        child.run_conversation('must not run')
    assert child.process.poll() is not None
    assert marker.read_bytes() == b''


def test_parent_control_loss_exits_worker_before_run(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    # Keep the Unix socket short independently of pytest's long temp path.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='ha-test-', dir='/tmp') as directory:
        listener = socket.socket(socket.AF_UNIX)
        endpoint = str(Path(directory) / 'control')
        listener.bind(endpoint)
        listener.listen(1)
        listener.settimeout(5)
        process = subprocess.Popen([sys.executable, '-m', 'agent.isolated_worker', endpoint],
            cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            process.stdin.write(('a'*64 + '\n').encode())
            process.stdin.close()
            peer, _ = listener.accept()
            peer.settimeout(5)
            assert receive(peer)['token'] == 'a'*64
            peer.close()  # Same EOF as a commissioning process exiting.
            # Main may finish before the daemon's orphan-exit callback runs;
            # either exit is terminal and no run was ever commissioned.
            assert process.wait(timeout=5) in {0, 70}
        finally:
            listener.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stderr.close()


def test_cron_claim_loss_cancels_pending_provider_admission(tmp_path, monkeypatch):
    from agent import workload_boundary
    from agent.workload_provider import WorkloadUnavailable
    from cron.scheduler import _run_agent_with_watchdog
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    entered, admission_cancelled, claim_lost = (threading.Event() for _ in range(3))
    class Provider:
        supported_kinds = {'cron-agent'}
        def popen(self, kind, args, *, config, cancel_event=None, **kwargs):
            entered.set()
            assert cancel_event.wait(10), 'provider never received cancellation'
            admission_cancelled.set()
            raise WorkloadUnavailable('admission cancelled')
    monkeypatch.setattr(workload_boundary, 'select_provider', lambda config: Provider())
    child = IsolatedAgent(kind='cron-agent')
    state = {}
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(_run_agent_with_watchdog, child, 'must not run', {},
            'claim-job', 'claim job', 'claim-task', claim_lost, state)
        try:
            assert entered.wait(3)
            claim_lost.set()
            with pytest.raises(RuntimeError, match='fire claim ownership'):
                future.result(timeout=8)
            assert admission_cancelled.wait(3)
            assert state['future'].result(timeout=3)['interrupted']
            assert child.process is None
        finally:
            child.interrupt()
