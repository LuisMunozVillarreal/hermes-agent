"""Commissioned profiles, not the launch process, own isolated worker state."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import pytest

from tests.agent.test_isolated_agent import isolated_home as _fixture_isolated_home, local_model as _fixture_local_model
isolated_home = _fixture_isolated_home
local_model = _fixture_local_model
from tests.agent.test_isolated_queue_integration import queue_home as _fixture_queue_home, wait_for as _fixture_wait_for, AGENT_CAP as _fixture_AGENT_CAP
queue_home = _fixture_queue_home
wait_for = _fixture_wait_for
AGENT_CAP = _fixture_AGENT_CAP


PROFILE_PLUGIN = '''
import json, os
from pathlib import Path
from hermes_constants import get_hermes_home
from agent.secret_scope import get_secret, is_multiplex_active
from hermes_cli.config_effective import load_user_config_effective
from tools.terminal_scope import terminal_env

def observation():
    return {
        'home': str(get_hermes_home()),
        'secret': get_secret('PROFILE_TEST_SECRET'),
        'foreign': get_secret('FOREIGN_ONLY_SECRET'),
        'multiplex': is_multiplex_active(),
        'identity': load_user_config_effective()['profile_test_identity'],
        'terminal_timeout': terminal_env('TERMINAL_TIMEOUT'),
        'pid': os.getpid(),
        'cgroup': Path('/proc/self/cgroup').read_text(),
    }

# Import-time reads must already be commissioned, not merely tool-time reads.
Path(get_hermes_home() / 'plugin-import.json').write_text(json.dumps(observation()))

def register(ctx):
    ctx.register_tool(name='profile_probe', toolset='profile_probe',
        schema={'name': 'profile_probe', 'description': 'Read fixture scope',
                'parameters': {'type': 'object', 'properties': {}}},
        handler=lambda args, **kw: json.dumps(observation()),
        check_fn=lambda: get_secret('PROFILE_TEST_SECRET') == 'synthetic-B')
'''


@pytest.fixture
def profile_model():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    class Requests(list):
        arguments = '{}'
        release = threading.Event()
    requests = Requests()
    requests.release.set()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if 'messages' not in body:
                self.send_error(400, 'fixture supports chat completions only')
                return
            requests.append(body)
            message = {'role': 'assistant', 'content': 'profile-verified'}
            finish = 'stop'
            if not any(m['role'] == 'tool' for m in body['messages']):
                if not requests.release.wait(30):
                    self.send_error(504)
                    return
                message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                    'index': 0, 'id': 'profile-call', 'type': 'function',
                    'function': {'name': 'profile_probe', 'arguments': requests.arguments}}]}
                finish = 'tool_calls'
            response = {'id': 'fixture', 'object': 'chat.completion', 'created': 1,
                'model': 'fixture', 'choices': [{'index': 0, 'finish_reason': finish,
                'message': message}], 'usage': {'prompt_tokens': 1,
                'completion_tokens': 1, 'total_tokens': 2}}
            if body.get('stream'):
                response['choices'][0]['delta'] = response['choices'][0].pop('message')
                payload = ('data: ' + json.dumps(response) + '\n\ndata: [DONE]\n\n').encode()
            else:
                payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream' if body.get('stream') else 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1', requests
    finally:
        requests.release.set()
        server.shutdown()
        server.server_close()
        thread.join(3)


# This observer runs the real worker and real AIAgent; only its observations are
# instrumented. A separate real-provider test below does not replace the launcher.
OBSERVER = '''
import sys, json
from pathlib import Path
import agent.isolated_worker as worker
original = worker.construct
def observe(*args, **kwargs):
    from hermes_constants import get_hermes_home
    from agent.secret_scope import get_secret, is_multiplex_active
    from hermes_cli.config_effective import load_user_config_effective
    from tools.terminal_scope import terminal_env
    agent = original(*args, **kwargs)
    Path(sys.argv[2]).write_text(json.dumps({
        'home': str(get_hermes_home()),
        'secret': get_secret('PROFILE_TEST_SECRET'),
        'foreign': get_secret('FOREIGN_ONLY_SECRET'),
        'multiplex': is_multiplex_active(),
        'identity': load_user_config_effective()['profile_test_identity'],
        'terminal_timeout': terminal_env('TERMINAL_TIMEOUT'),
    }))
    return agent
worker.construct = observe
worker.main(sys.argv[1])
'''


def session_ids(home):
    path = home / 'state.db'
    if not path.exists():
        return []
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        return [row[0] for row in db.execute('SELECT id FROM sessions')]


@pytest.mark.platforms("linux")
def test_commissioned_profile_survives_scope_exit(local_model, isolated_home, monkeypatch):
    from agent.isolated_agent import IsolatedAgent
    from agent.secret_scope import (set_secret_scope, reset_secret_scope,
                                    set_multiplex_active, is_multiplex_active)
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from tools.terminal_scope import set_terminal_scope, reset_terminal_scope

    a = isolated_home
    b = a / 'profile-B'
    b.mkdir()
    (a / 'config.yaml').write_text('profile_test_identity: A\ncompression:\n  enabled: false\n')
    (b / 'config.yaml').write_text('profile_test_identity: B\ncompression:\n  enabled: false\n')
    monkeypatch.setenv('PROFILE_TEST_SECRET', 'synthetic-A')
    monkeypatch.setenv('FOREIGN_ONLY_SECRET', 'foreign-A')
    monkeypatch.setenv('TERMINAL_TIMEOUT', '111')
    monkeypatch.setenv('HERMES_PROFILE', 'untrusted-ambient-label')
    output = a / 'observed.json'
    launch_envs = []

    def launch(args, **options):
        launch_envs.append(dict(options['env']))
        return subprocess.Popen([args[0], '-c', OBSERVER, args[-1], str(output)], **options)

    previous = is_multiplex_active()
    set_multiplex_active(True)
    home = set_hermes_home_override(str(b))
    secrets = {'PROFILE_TEST_SECRET': 'synthetic-B'}
    secret = set_secret_scope(secrets)
    terminal = set_terminal_scope({'TERMINAL_TIMEOUT': '222', 'TERMINAL_ENV': 'local'})
    try:
        child = IsolatedAgent(kind='delegation', launcher=launch,
            model='fixture', provider='custom', base_url=local_model[0],
            enabled_toolsets=[], skip_memory=True, skip_context_files=True, quiet_mode=True)
    finally:
        reset_terminal_scope(terminal)
        reset_secret_scope(secret)
        reset_hermes_home_override(home)
        set_multiplex_active(previous)
    secrets['PROFILE_TEST_SECRET'] = 'changed-after-commission'
    result = child.run_conversation('brief fixture result')
    assert result['completed']
    assert json.loads(output.read_text()) == {
        'home': str(b), 'secret': 'synthetic-B', 'foreign': None,
        'multiplex': True, 'identity': 'B', 'terminal_timeout': '222',
    }
    assert child.session_id in session_ids(b)
    assert child.session_id not in session_ids(a)
    assert launch_envs[0]['HERMES_HOME'] == str(b)
    assert 'synthetic-A' not in launch_envs[0].values()
    assert 'synthetic-B' not in launch_envs[0].values()
    assert 'foreign-A' not in launch_envs[0].values()
    assert launch_envs[0].get('HERMES_PROFILE') != 'untrusted-ambient-label'


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('kind', ['delegation', 'cron-agent'])
def test_real_queue_profile_plugin_tool_and_session(queue_home, profile_model, monkeypatch, kind):
    from agent.isolated_agent import IsolatedAgent
    from agent.secret_scope import (set_secret_scope, reset_secret_scope,
                                    set_multiplex_active, is_multiplex_active)
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    b, _state, client, _provider, backend = queue_home
    a = b / 'ambient-A'
    a.mkdir()
    (a / 'config.yaml').write_text('profile_test_identity: A\n')
    plugin = b / 'plugins' / 'profile-probe'
    plugin.mkdir()
    (plugin / 'plugin.yaml').write_text('name: profile-probe\nversion: 1.0.0\ndescription: fixture\n')
    (plugin / '__init__.py').write_text(PROFILE_PLUGIN)
    config = b / 'config.yaml'
    config.write_text(config.read_text().replace('[hermes-worker-queue]',
        '[hermes-worker-queue, profile-probe]') +
        'profile_test_identity: B\nterminal:\n  timeout: 222\ntools:\n  tool_search: false\n')
    monkeypatch.setenv('HERMES_HOME', str(a))
    monkeypatch.setenv('PROFILE_TEST_SECRET', 'synthetic-A')
    monkeypatch.setenv('FOREIGN_ONLY_SECRET', 'foreign-A')
    monkeypatch.setenv('TERMINAL_TIMEOUT', '111')
    home = set_hermes_home_override(str(b))
    secret = set_secret_scope({'PROFILE_TEST_SECRET': 'synthetic-B'})
    previous = is_multiplex_active()
    set_multiplex_active(True)
    child = None
    try:
        child = IsolatedAgent(kind=kind, model='fixture', provider='custom',
            base_url=profile_model[0], enabled_toolsets=['profile_probe'],
            max_iterations=3, skip_memory=True, skip_context_files=True, quiet_mode=True)
        result = child.run_conversation('call profile_probe')
        assert result['completed'], result
        imported = json.loads((b / 'plugin-import.json').read_text())
        tools = [m for r in profile_model[1] for m in r['messages'] if m['role'] == 'tool']
        assert tools, result
        assert tools[0]['content'].startswith('{'), tools
        observed = json.loads(tools[0]['content'])
        for evidence in (imported, observed):
            assert evidence['home'] == str(b)
            assert evidence['identity'] == 'B'
            assert evidence['secret'] == 'synthetic-B'
            assert evidence['foreign'] is None
            assert evidence['multiplex'] is True
            assert evidence['terminal_timeout'] == '222'
            assert evidence['pid'] == child.worker_pid != os.getpid()
        assert child.session_id in session_ids(b)
        assert not session_ids(a)
        assert not (a / 'plugin-import.json').exists()
        wait_for(lambda: client.call('status')['reserved'] == 0)
        status = client.call('status')
        assert len(status['jobs']) == 1
        row = status['jobs'][0]
        assert row['state'] == 'finished' and row['kind'] == kind
        assert row['peak'] == AGENT_CAP
        assert row['unit'] in imported['cgroup']
        assert backend.empty(row['unit'])
        assert 'synthetic-' not in json.dumps(status)
        print(json.dumps({'kind': kind, 'profile': 'B', 'plugin_import': 'B',
            'tool_secret_scope': 'B', 'foreign_secret': None, 'multiplex': True,
            'session_in_B_only': True, 'reserved': status['reserved'],
            'worker_unit_empty': True, 'worker_state': row['state']}))
    finally:
        if child:
            child.close()
        set_multiplex_active(previous)
        reset_secret_scope(secret)
        reset_hermes_home_override(home)
        # The imported queue fixture owns registration under B.
        monkeypatch.setenv('HERMES_HOME', str(b))


@pytest.mark.platforms("linux")
def test_default_home_snapshot_survives_ambient_rebind(queue_home, local_model, monkeypatch):
    from agent.isolated_agent import IsolatedAgent

    home, _state, client, _provider, backend = queue_home
    child = IsolatedAgent(kind='delegation', model='fixture', provider='custom',
        base_url=local_model[0], enabled_toolsets=[], skip_memory=True,
        skip_context_files=True, quiet_mode=True)
    foreign = home / 'later-ambient-home'
    foreign.mkdir()
    (foreign / 'config.yaml').write_text('workloads:\n  mode: required\n  provider: missing\n')
    monkeypatch.setenv('HERMES_HOME', str(foreign))
    try:
        result = child.run_conversation('retain commissioned default profile')
        assert result['completed'], result
        assert child.session_id in session_ids(home)
        assert not session_ids(foreign)
        wait_for(lambda: client.call('status')['reserved'] == 0)
        rows = client.call('status')['jobs']
        assert len(rows) == 1 and rows[0]['state'] == 'finished'
        assert backend.empty(rows[0]['unit'])
    finally:
        child.close()
        monkeypatch.setenv('HERMES_HOME', str(home))
