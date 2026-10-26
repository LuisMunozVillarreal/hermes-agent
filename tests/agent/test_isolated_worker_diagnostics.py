"""Failure reports from real subprocesses, without a native queue provider."""
from concurrent.futures import ThreadPoolExecutor
import logging
import re
import subprocess
import sys
import threading

import pytest

from agent.isolated_agent import IsolatedAgent

pytestmark = pytest.mark.platforms('linux')


@pytest.mark.parametrize('outcome, expected', [
    ('bootstrap', 'exit_code=23 state=exited'),
    ('run', 'exit_code=23 state=exited'),
    ('signal', 'exit_code=-15 state=exited'),
    ('alive', 'exit_code=unknown state=running'),
])
def test_worker_exit_reports_status_pid_and_safe_correlation(outcome, expected, caplog):
    phase = 'bootstrap' if outcome == 'bootstrap' else 'run'
    code = 'import sys; sys.stdin.readline(); sys.exit(23)'
    if phase == 'run':
        code = (
            'import socket,sys; from agent.isolated_protocol import send; '
            'from agent.isolated_transport import receive_message; '
            'token=sys.stdin.readline().strip(); s=socket.socket(socket.AF_UNIX); '
            "s.connect(sys.argv[1]); send(s, {'v':1,'token':token}); "
            'receive_message(s, token); sys.stderr.write("private-stderr-secret\\n"); '
            'sys.exit(23)'
        )
        if outcome == 'signal':
            code = code.replace('sys.exit(23)', 'import os,signal; os.kill(os.getpid(), signal.SIGTERM)')
        elif outcome == 'alive':
            code = code.replace('sys.exit(23)', 's.close(); import threading; threading.Event().wait(30)')
    def launch(args, **options):
        return subprocess.Popen([sys.executable, '-c', code, args[-1]], **options)
    child = IsolatedAgent(kind='delegation', launcher=launch, api_key='private-api-key',
                          session_id='private-session')
    with caplog.at_level(logging.WARNING, logger='agent.isolated_agent'):
        with pytest.raises(RuntimeError):
            child.run_conversation('private-prompt')
    records = [r.message for r in caplog.records if r.name == 'agent.isolated_agent']
    assert len(records) == 1
    report = records[0]
    assert f'phase={phase}' in report
    if phase == 'bootstrap':
        assert 'agent worker did not connect' in report
    assert expected in report
    assert f'worker_pid={child.process.pid}' in report
    assert re.search(r'diagnostic_id=[0-9a-f]{32}\b', report)
    assert 'private-' not in report
    assert len(report) < 512
    assert child.process.poll() is not None


def test_successful_reply_and_normal_worker_exit_are_quiet(caplog):
    code = (
        'import socket,sys; from agent.isolated_protocol import send; '
        'from agent.isolated_transport import receive_message, send_message; import time; '
        'token=sys.stdin.readline().strip(); s=socket.socket(socket.AF_UNIX); '
        "s.connect(sys.argv[1]); send(s, {'v':1,'token':token}); "
        'request=receive_message(s, token); '
        "send_message(s, {'v':1,'generation':token,'reply':request['id'], "
        "'value':{'final_response':'done','messages':[]}}, time.monotonic()+3)"
    )
    def launch(args, **options):
        return subprocess.Popen([sys.executable, '-c', code, args[-1]], **options)
    child = IsolatedAgent(kind='delegation', launcher=launch)
    assert child.run_conversation('private-prompt')['final_response'] == 'done'
    assert child.process.poll() == 0
    assert not [r for r in caplog.records if r.name == 'agent.isolated_agent']


def test_cancelled_bootstrap_emits_no_failure_report(caplog):
    launched = threading.Event()
    # Hold the child independently of stdin EOF until the parent terminates it.
    def blocked_launch(args, **options):
        process = subprocess.Popen(
            [sys.executable, '-c', 'import threading; threading.Event().wait(30)'], **options)
        launched.set()
        return process
    child = IsolatedAgent(kind='delegation', launcher=blocked_launch)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(child.run_conversation, 'private-prompt')
        try:
            assert launched.wait(3)
            child.interrupt()
            assert future.result(timeout=5)['interrupted']
        finally:
            child.interrupt()
            child.close()
    assert not [r for r in caplog.records if r.name == 'agent.isolated_agent']


def test_cancellation_during_exit_observation_remains_quiet(monkeypatch, caplog):
    from agent import isolated_agent
    observing, release = threading.Event(), threading.Event()
    original = isolated_agent.worker_status
    def observe(process):
        observing.set()
        assert release.wait(3)
        return original(process)
    monkeypatch.setattr(isolated_agent, 'worker_status', observe)
    def launch(args, **options):
        return subprocess.Popen(
            [sys.executable, '-c', 'import sys; sys.stdin.readline(); sys.exit(23)'], **options)
    child = IsolatedAgent(kind='delegation', launcher=launch)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(child.run_conversation, 'private-prompt')
        try:
            assert observing.wait(3)
            child.interrupt()
            release.set()
            assert future.result(timeout=3)['interrupted']
        finally:
            release.set()
            child.interrupt()
            child.close()
    assert not [r for r in caplog.records if r.name == 'agent.isolated_agent']
