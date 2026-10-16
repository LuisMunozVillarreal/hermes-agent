"""Native AIAgent over private IPC, using a credential-free local model."""
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.fixture
def local_model():
    class Requests(list):
        command = None
        authorization = None
        response_text = 'isolated-result'
    requests = Requests()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            requests.authorization = self.headers.get('Authorization')
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append(body)
            response = {'id': 'fixture', 'object': 'chat.completion', 'created': 1,
                        'model': 'fixture', 'choices': [{'index': 0, 'finish_reason': 'stop',
                        'message': {'role': 'assistant', 'content': requests.response_text}}],
                        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}
            if requests.command and body.get('messages') and not any(m['role'] == 'tool' for m in body['messages']):
                response['choices'][0] = {'index': 0, 'finish_reason': 'tool_calls', 'message': {
                    'role': 'assistant', 'content': None, 'tool_calls': [{'index': 0, 'id': 'call_fixture',
                    'type': 'function', 'function': {'name': 'terminal', 'arguments': json.dumps({
                        'command': requests.command, 'timeout': 5})}}]}}
            if body.get('stream'):
                response['object'] = 'chat.completion.chunk'
                choice = response['choices'][0]
                choice['delta'] = choice.pop('message')
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
        server.shutdown()
        server.server_close()
        thread.join(3)


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('TERMINAL_CWD', str(tmp_path))
    monkeypatch.delenv('HERMES_IGNORE_USER_CONFIG', raising=False)
    (tmp_path / 'config.yaml').write_text('model:\n  default: fixture\ncompression:\n  enabled: false\n')
    return tmp_path


def test_native_agent_process_result(local_model, isolated_home):
    from agent.isolated_agent import IsolatedAgent

    url, requests = local_model
    child = IsolatedAgent(kind='delegation', launcher=subprocess.Popen,
                          model='fixture', provider='custom', base_url=url,
                          enabled_toolsets=[], skip_memory=True, skip_context_files=True,
                          max_iterations=3, quiet_mode=True)
    try:
        result = child.run_conversation('answer briefly', task_id='native-fixture')
        assert result['final_response'] == 'isolated-result', result
        assert child.worker_pid != os.getpid()
        assert requests
        assert child.session_id
    finally:
        child.close()
    assert child.process.poll() is not None


def test_native_large_history_and_result_roundtrip(local_model, isolated_home):
    from agent.isolated_agent import IsolatedAgent
    from agent.isolated_protocol import MAX_FRAME
    url, requests = local_model
    import random
    import string
    # Transport size, not degenerate model output: current agents reject runs
    # of one repeated character before the result reaches the IPC boundary.
    rng = random.Random(0)
    alphabet = string.ascii_letters + ' '
    history_text = ''.join(rng.choices(alphabet, k=MAX_FRAME + 128))
    requests.response_text = ''.join(rng.choices(alphabet, k=MAX_FRAME + 256))
    history = [{'role': 'user', 'content': history_text},
               {'role': 'assistant', 'content': 'remembered'}]
    child = IsolatedAgent(kind='delegation', launcher=subprocess.Popen,
        model='fixture', provider='custom', base_url=url, enabled_toolsets=[],
        skip_memory=True, skip_context_files=True, quiet_mode=True)
    streamed = []
    result = child.run_conversation('retain exact history', conversation_history=history,
        stream_callback=streamed.append)
    assert result['completed'], {k: v for k, v in result.items() if k not in {'messages', 'final_response'}}
    assert result['final_response'] == requests.response_text
    assert ''.join(streamed) == requests.response_text
    assert any(m.get('content') == history_text for r in requests for m in r.get('messages', []))
    assert any(m.get('content') == history_text for m in result['messages'])
    assert child.process.poll() is not None


def test_stream_history_and_schema_retry_fidelity(local_model, isolated_home):
    from agent.isolated_agent import IsolatedAgent
    from tools.delegate_tool_child_run import _validate_child_output_schema
    url, requests = local_model
    child = IsolatedAgent(kind='delegation', launcher=subprocess.Popen,
        model='fixture', provider='custom', base_url=url, enabled_toolsets=[],
        skip_memory=True, skip_context_files=True, max_iterations=3, quiet_mode=True)
    child._delegate_output_schema = {'type': 'object'}
    streams = []
    history = [{'role': 'user', 'content': 'EXPLICIT_HISTORY'},
               {'role': 'assistant', 'content': 'remembered'}]
    first = child.run_conversation('FIRST_GOAL', task_id='retry-history',
        stream_callback=streams.append, conversation_history=history)
    assert ''.join(streams) == first['final_response']
    first_request = next(r for r in requests if r.get('stream') is True)
    assert any(m.get('content') == 'EXPLICIT_HISTORY' for m in first_request['messages'])
    requests.clear()
    streams.clear()
    outcome = _validate_child_output_schema(child, first, 0, 'retry-history', streams.append)
    assert outcome.retries == 1 and streams
    retry_request = next(r for r in requests if r.get('stream') is True)
    assert any(m.get('content') == 'FIRST_GOAL' for m in retry_request['messages'])
    assert retry_request['messages'][:len(first_request['messages'])] == first_request['messages']


def test_runtime_credential_and_progress_fidelity(local_model, isolated_home):
    from agent.isolated_agent import IsolatedAgent
    url, requests = local_model
    requests.command = 'printf fixture-output'
    events = []
    child = IsolatedAgent(kind='delegation', launcher=subprocess.Popen,
        model='fixture', provider='custom', base_url=url, api_key='synthetic-runtime-key',
        tool_progress_callback=lambda *a, **kw: events.append((a, kw)),
        enabled_toolsets=['terminal'], skip_memory=True, skip_context_files=True,
        max_iterations=3, quiet_mode=True)
    result = child.run_conversation('execute fixture')
    assert result['final_response'] == 'isolated-result'
    assert requests.authorization == 'Bearer synthetic-runtime-key'
    assert any('terminal' in str(event) for event in events)
    assert child.get_activity_summary()['api_call_count'] >= 1


@pytest.mark.parametrize('choice', ['once', 'deny'])
def test_native_approval_and_steering(local_model, isolated_home, choice):
    from agent.isolated_agent import IsolatedAgent
    url, requests = local_model
    marker = isolated_home / 'approved'
    requests.command = f'rm -rf {isolated_home}/nonexistent; touch {marker}'
    approvals = []
    def approve(command, description, **kwargs):
        approvals.append(command)
        assert child.steer('Use the corrected steering instruction')
        return choice
    child = IsolatedAgent(kind='delegation', launcher=subprocess.Popen,
                          approval_callback=approve, model='fixture', provider='custom', base_url=url,
                          enabled_toolsets=['terminal'], skip_memory=True, skip_context_files=True,
                          max_iterations=3, quiet_mode=True)
    result = child.run_conversation('execute the fixture command', task_id='approval-fixture')
    assert approvals, result
    assert marker.exists() == (choice == 'once'), result
    assert any('corrected steering instruction' in str(r.get('messages')) for r in requests)
    assert result['final_response'] == 'isolated-result'
    assert not child.steer('too late')


def test_late_old_worker_metrics_cannot_mutate_next_turn(local_model, isolated_home):
    from agent.isolated_agent import IsolatedAgent
    from concurrent.futures import ThreadPoolExecutor
    url, _ = local_model
    waiting, release, stale_done = (threading.Event() for _ in range(3))
    class DelayedMetrics(IsolatedAgent):
        delayed = False
        def _handle_event(self, op, payload):
            if op == 'metrics' and not self.delayed:
                self.delayed = True
                waiting.set()
                assert release.wait(10)
                try:
                    return super()._handle_event(op, dict(payload, model='stale-model'))
                finally:
                    stale_done.set()
            return super()._handle_event(op, payload)
    child = DelayedMetrics(kind='delegation', launcher=subprocess.Popen,
        model='fixture', provider='custom', base_url=url, enabled_toolsets=[],
        skip_memory=True, skip_context_files=True, quiet_mode=True)
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(child.run_conversation, 'first')
        try:
            assert waiting.wait(10)
            with pytest.raises(RuntimeError, match='already running'):
                child.run_conversation('concurrent turn is forbidden')
            child._channel.close()
            with pytest.raises(RuntimeError, match='connection lost'):
                first.result(timeout=8)
            def launch(args, **kwargs):
                release.set()
                assert stale_done.wait(3)
                assert child.model == 'fixture'
                return subprocess.Popen(args, **kwargs)
            child._launcher = launch
            assert child.run_conversation('new turn')['completed']
        finally:
            release.set()
            child.interrupt()
            child.close()


def test_schema_retry_cancel_fences_relaunch(local_model, isolated_home):
    from agent.isolated_agent import IsolatedAgent
    from tools.delegate_tool_child_run import _validate_child_output_schema
    url, requests = local_model
    child = IsolatedAgent(kind='delegation', launcher=subprocess.Popen,
        model='fixture', provider='custom', base_url=url, enabled_toolsets=['terminal'],
        skip_memory=True, skip_context_files=True, max_iterations=3, quiet_mode=True)
    child._delegate_output_schema = {'type': 'object'}
    first = child.run_conversation('FIRST_GOAL', task_id='retry-cancel')
    marker = isolated_home / 'retry-side-effect'
    requests.command = f'touch {marker}'
    requests.clear()
    interrupted = []
    def launch(args, **kwargs):
        interrupted.append(child.interrupt('cancel schema retry'))
        return subprocess.Popen(args, **kwargs)
    child._launcher = launch
    outcome = _validate_child_output_schema(child, first, 0, 'retry-cancel', None)
    assert interrupted == [True]
    assert outcome.retries == 1
    assert not marker.exists() and not requests
    assert child.run_conversation('never resume')['interrupted']
    assert child.process.poll() is not None


def test_cancel_before_launcher_returns_never_bootstraps(local_model, isolated_home):
    from agent.isolated_agent import IsolatedAgent
    url, requests = local_model
    marker = isolated_home / 'cancelled-side-effect'
    requests.command = f'touch {marker}'
    child = IsolatedAgent(kind='delegation', model='fixture', provider='custom', base_url=url,
        enabled_toolsets=['terminal'], skip_memory=True, skip_context_files=True, quiet_mode=True)
    def launch(args, **kwargs):
        assert child.interrupt('cancel pending admission')
        return subprocess.Popen(args, **kwargs)
    child._launcher = launch
    result = child.run_conversation('execute fixture', task_id='cancel-pending')
    assert result.get('interrupted') and not result.get('completed'), result
    assert not marker.exists()
    assert not requests
    assert child.process.poll() is not None


@pytest.mark.parametrize('stop', ['interrupt', 'disconnect'])
def test_cancel_during_approval_rejects_late_allow(local_model, isolated_home, stop):
    from agent.isolated_agent import IsolatedAgent
    from concurrent.futures import ThreadPoolExecutor
    url, requests = local_model
    marker = isolated_home / 'must-not-run'
    requests.command = f'rm -rf {isolated_home}/nonexistent; touch {marker}'
    asked, release = threading.Event(), threading.Event()
    def approve(*args, **kwargs):
        asked.set()
        release.wait(15)
        return 'once'
    child = IsolatedAgent(kind='delegation', launcher=subprocess.Popen,
                          approval_callback=approve, model='fixture', provider='custom', base_url=url,
                          enabled_toolsets=['terminal'], skip_memory=True, skip_context_files=True,
                          max_iterations=3, quiet_mode=True)
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(child.run_conversation, 'execute fixture', task_id='cancel-fixture')
        try:
            assert asked.wait(15)
            if stop == 'interrupt':
                child.interrupt('cancel test')
                result = future.result(timeout=8)
                assert result['interrupted'] is True
                assert not result.get('completed')
            else:
                child._channel.close()  # Authority disappears; no terminate() call.
                with pytest.raises(RuntimeError, match='connection lost'):
                    future.result(timeout=8)
                assert child.process.returncode == 70
        finally:
            release.set()
            child.close()
    assert not marker.exists()
    assert child.process.poll() is not None
    assert not child.steer('stale steering')


@pytest.fixture
def native_provider(isolated_home):
    from agent.workload_registry import register_provider, restore_registration
    from agent.workload_provider import WorkloadProvider
    class Provider(WorkloadProvider):
        name = 'isolated-fixture'
        supported_kinds = frozenset({'delegation', 'cron-agent'})
        def submit(self, spec):
            raise AssertionError('no advisory launch')
        def popen(self, kind, args, *, config, **kwargs):
            process = subprocess.Popen(args, **kwargs)
            process.workload_lease = {'ticket': 'a'*32, 'generation': 'b'*32}
            return process
    provider = Provider()
    with (isolated_home / 'config.yaml').open('a') as f:
        f.write('workloads:\n  mode: required\n  provider: isolated-fixture\n')
    register_provider(provider)
    try:
        yield provider
    finally:
        restore_registration(provider.name, provider, None)


def test_native_delegation_front_door(local_model, isolated_home, native_provider):
    from run_agent import AIAgent
    from tools.delegate_tool import delegate_task
    url, _requests = local_model
    parent = AIAgent(model='fixture', provider='custom', base_url=url, api_key='not-transferred',
                     enabled_toolsets=[], skip_memory=True, skip_context_files=True, quiet_mode=True)
    try:
        result = json.loads(delegate_task(goal='answer briefly', parent_agent=parent))
        assert 'isolated-result' in str(result), result
        assert not result.get('error'), result
    finally:
        parent.close()


def test_background_batch_durable_completion_and_fenced_delivery(local_model, isolated_home, native_provider):
    import queue
    from run_agent import AIAgent
    from tools.delegate_tool import delegate_task
    from tools import async_delegation as ad
    from tools.process_registry import process_registry
    from gateway.session_context import set_session_vars, clear_session_vars
    ad._reset_for_tests()
    url, _requests = local_model
    parent = AIAgent(model='fixture', provider='custom', base_url=url, api_key='batch-synthetic',
        session_id='batch-owner', enabled_toolsets=[], skip_memory=True, skip_context_files=True, quiet_mode=True)
    tokens = set_session_vars(platform='api_server', chat_id='batch-owner',
        session_key='batch-owner', session_id='batch-owner', session_history_delivery='1', async_delivery=False)
    try:
        dispatch = json.loads(delegate_task(tasks=[{'goal': 'first brief answer'}, {'goal': 'second brief answer'}],
            background=True, parent_agent=parent))
        assert dispatch.get('status') == 'dispatched', dispatch
        event = process_registry.completion_queue.get(timeout=25)
        assert event['delegation_id'] == dispatch['delegation_id']
        assert len(event['results']) == 2
        assert all(r['status'] == 'completed' for r in event['results']), event
        assert all(r['tokens']['input'] > 0 for r in event['results']), event
        restored = queue.Queue()
        assert ad.restore_undelivered_completions(restored) == 1
        saved = restored.get_nowait()
        assert saved['results'] == event['results']
        assert saved['restored'] is True
        assert saved['origin_session_id'] == 'batch-owner'
        key = dispatch['delegation_id']
        assert ad.claim_completion_delivery(key, 'current-generation')
        assert not ad.complete_completion_delivery(key, 'stale-generation')
        assert ad.complete_completion_delivery(key, 'current-generation')
        assert ad.restore_undelivered_completions(queue.Queue()) == 0
    finally:
        parent.close()
        clear_session_vars(tokens)
        ad._reset_for_tests()


def test_native_cron_agent_front_door(local_model, isolated_home, native_provider):
    from cron.scheduler import run_job
    url, _requests = local_model
    config = isolated_home / 'config.yaml'
    config.write_text(config.read_text() + f'cron:\n  model: fixture\n  provider: custom\n  base_url: {url}\n  api_key: local-fixture\n')
    result = run_job({'id': 'native-fixture', 'name': 'native fixture', 'prompt': 'answer briefly',
                      'model': 'fixture', 'provider': 'custom', 'base_url': url})
    assert result[0] is True, result
    assert result[2] == 'isolated-result', result


@pytest.fixture
def process_provider(monkeypatch):
    """Exercise the native process boundary, NOT kernel/resource enforcement."""
    from agent import workload_boundary
    class Provider:
        supported_kinds = frozenset({'delegation', 'cron-agent'})
        def popen(self, kind, args, *, config, **kwargs):
            process = subprocess.Popen(args, **kwargs)
            process.workload_lease = {'ticket': 'a'*32, 'generation': 'b'*32}
            return process
    monkeypatch.setattr(workload_boundary, 'select_provider', lambda config: Provider())


def test_cron_worker_inherits_trusted_delivery_context(local_model, isolated_home, monkeypatch):
    from agent import workload_boundary
    from cron.scheduler import run_job
    url, _ = local_model
    observation = isolated_home / 'worker-routing.json'
    code = '''import sys,json
from pathlib import Path
import agent.isolated_worker as worker
from tools.send_message_tool import _maybe_skip_cron_duplicate_send
from tools.cronjob_job_args import _resolve_cron_context_deliver
from gateway.session_context import async_delivery_supported, session_history_delivery_supported
original = worker.construct
def observed(*args, **kwargs):
    agent = original(*args, **kwargs)
    Path(sys.argv[2]).write_text(json.dumps({
        'duplicate': _maybe_skip_cron_duplicate_send('telegram', '123', '456'),
        'nested_origin': _resolve_cron_context_deliver('origin'),
        'history_delivery': session_history_delivery_supported()}))
    return agent
worker.construct = observed
worker.main(sys.argv[1])
'''
    class Provider:
        supported_kinds = {'cron-agent'}
        def popen(self, kind, args, *, config, **opts):
            process = subprocess.Popen([args[0], '-c', code, args[-1], str(observation)], **opts)
            process.workload_lease = {'ticket': 'a'*32, 'generation': 'b'*32}
            return process
    monkeypatch.setattr(workload_boundary, 'select_provider', lambda config: Provider())
    monkeypatch.setenv('TELEGRAM_BOT_TOKEN', 'synthetic-unused-no-network')
    with (isolated_home / 'config.yaml').open('a') as config:
        config.write(f'cron:\n  model: fixture\n  provider: custom\n  base_url: {url}\n  api_key: fixture\n')
    result = run_job({'id': 'routing', 'name': 'routing', 'prompt': 'answer briefly',
        'model': 'fixture', 'provider': 'custom', 'base_url': url, 'deliver': 'telegram:123:456'})
    assert result[0], result
    observed = json.loads(observation.read_text())
    assert observed['duplicate']['skipped'] is True
    assert observed['nested_origin'] == 'telegram:123:456'
    assert observed['history_delivery'] is False


@pytest.mark.parametrize('mode', ['deny', 'approve'])
def test_cron_policy_precedes_delegation_autoapprove(local_model, isolated_home, process_provider, mode):
    from cron.scheduler import run_job
    url, requests = local_model
    marker = isolated_home / 'cron-operation'
    requests.command = f'rm -rf {isolated_home}/nonexistent; touch {marker}'
    with (isolated_home / 'config.yaml').open('a') as config:
        config.write(f'approvals:\n  mode: manual\n  cron_mode: {mode}\n'
                     'delegation:\n  subagent_auto_approve: true\n'
                     f'cron:\n  model: fixture\n  provider: custom\n  base_url: {url}\n  api_key: fixture\n')
    result = run_job({'id': 'cron-policy', 'name': 'cron policy', 'prompt': 'execute fixture',
                      'model': 'fixture', 'provider': 'custom', 'base_url': url})
    assert result[0], result
    assert marker.exists() == (mode == 'approve'), [m for r in requests for m in r.get('messages', []) if m['role'] == 'tool']


@pytest.mark.parametrize('surface', ['api_server', 'cron', 'single_query'])
def test_unattended_delegation_never_uses_interactive_allow(local_model, isolated_home, monkeypatch, surface):
    from agent.isolated_agent import IsolatedAgent
    from gateway.session_context import set_session_vars, clear_session_vars
    from agent.isolated_approval import decide
    url, requests = local_model
    marker = isolated_home / 'unattended-denied'
    requests.command = f'rm -rf {isolated_home}/nonexistent; touch {marker}'
    with (isolated_home / 'config.yaml').open('a') as config:
        config.write('approvals:\n  mode: manual\n  unattended_mode: deny\n'
                     '  cron_mode: deny\n  single_query_mode: deny\n'
                     'delegation:\n  subagent_auto_approve: true\n')
    monkeypatch.setenv('HERMES_SINGLE_QUERY_SESSION', '1' if surface == 'single_query' else '')
    tokens = set_session_vars(platform='api_server' if surface == 'api_server' else '',
                              cron_session='1' if surface == 'cron' else '')
    callbacks = []
    def allow(*args, **kwargs):
        callbacks.append(args)
        return 'once'
    try:
        # Non-terminal approval relays must also apply policy before callbacks.
        assert decide({'command': 'operation', 'description': 'dangerous'}, allow) == 'deny'
        child = IsolatedAgent(kind='delegation', launcher=subprocess.Popen,
            approval_callback=allow, model='fixture', provider='custom', base_url=url,
            enabled_toolsets=['terminal'], skip_memory=True, skip_context_files=True, quiet_mode=True)
        result = child.run_conversation('execute fixture')
        assert result['completed'], result
        assert not marker.exists()
        assert not callbacks
    finally:
        clear_session_vars(tokens)


@pytest.mark.parametrize('surface', ['cron', 'delegation', 'worktree'])
def test_native_task_directory_overrides_ambient(local_model, isolated_home, process_provider, surface):
    from pathlib import Path
    from tools.terminal_tool import record_session_cwd, clear_session_cwd
    url, requests = local_model
    intended = isolated_home / 'project'
    intended.mkdir()
    (intended / 'AGENTS.md').write_text('Workspace instruction: PROJECT_CONTEXT_SENTINEL.')
    requests.command = 'pwd'
    if surface == 'cron':
        from cron.scheduler import run_job
        with (isolated_home / 'config.yaml').open('a') as config:
            config.write(f'cron:\n  model: fixture\n  provider: custom\n  base_url: {url}\n  api_key: fixture\n')
        result = run_job({'id': 'cwd', 'name': 'cwd', 'prompt': 'execute fixture',
            'model': 'fixture', 'provider': 'custom', 'base_url': url, 'workdir': str(intended)})
        assert result[0], result
    else:
        from run_agent import AIAgent
        from tools.delegate_tool import delegate_task
        if surface == 'worktree':
            subprocess.run(['git', 'init', '-q', str(intended)], check=True)
            subprocess.run(['git', '-C', str(intended), 'add', 'AGENTS.md'], check=True)
            subprocess.run(['git', '-C', str(intended), '-c', 'user.name=Fixture',
                '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
            with (isolated_home / 'config.yaml').open('a') as config:
                config.write('delegation:\n  worktree_isolation: true\n')
        parent = AIAgent(model='fixture', provider='custom', base_url=url, api_key='fixture',
            enabled_toolsets=['terminal'], skip_memory=True, skip_context_files=True, quiet_mode=True)
        parent._current_task_id = 'cwd-parent'
        record_session_cwd(parent._current_task_id, str(intended))
        try:
            result = json.loads(delegate_task(goal='execute fixture', parent_agent=parent))
            assert not result.get('error'), result
        finally:
            clear_session_cwd(parent._current_task_id)
            parent.close()
    tools = [m for r in requests for m in r.get('messages', []) if m['role'] == 'tool']
    assert tools, result
    tool_result = json.loads(tools[-1]['content'])
    assert tool_result['exit_code'] == 0, tool_result
    actual = Path(tool_result['output'].strip())
    if surface == 'worktree':
        assert actual != intended and intended in actual.parents, (actual, result)
    else:
        assert actual == intended, tool_result
    if surface == 'cron':
        assert any('PROJECT_CONTEXT_SENTINEL' in m.get('content', '')
            for r in requests for m in r.get('messages', []) if m['role'] == 'system')


def test_finalization_never_acknowledges_lost_steering(local_model, isolated_home):
    from agent.isolated_agent import IsolatedAgent
    url, _ = local_model
    accepted = []
    class RacingAgent(IsolatedAgent):
        def _handle_event(self, op, payload):
            if op == 'activity':
                accepted.append(self.steer('exact late steering'))
            return super()._handle_event(op, payload)
    child = RacingAgent(kind='delegation', launcher=subprocess.Popen,
        model='fixture', provider='custom', base_url=url,
        enabled_toolsets=[], skip_memory=True, skip_context_files=True, quiet_mode=True)
    result = child.run_conversation('brief response')
    assert accepted
    assert not accepted[0] or 'exact late steering' in result.get('pending_steer', '')


def test_gateway_approval_keeps_parent_authority(local_model, isolated_home):
    from agent.isolated_agent import IsolatedAgent
    from tools import approval
    from gateway.session_context import set_session_vars, clear_session_vars
    url, requests = local_model
    marker = isolated_home / 'gateway-approved'
    requests.command = f'rm -rf {isolated_home}/nonexistent; touch {marker}'
    notifications = []
    def notify(data):
        notifications.append(data)
        approval.resolve_gateway_approval('gateway-owner', 'once')
    tokens = set_session_vars(platform='telegram', session_key='gateway-owner')
    approval.register_gateway_notify('gateway-owner', notify)
    try:
        child = IsolatedAgent(kind='delegation', launcher=subprocess.Popen,
                              model='fixture', provider='custom', base_url=url,
                              enabled_toolsets=['terminal'], skip_memory=True, skip_context_files=True,
                              max_iterations=3, quiet_mode=True)
        result = child.run_conversation('execute fixture', task_id='gateway-fixture')
        assert notifications, result
        assert marker.exists(), result
        assert not approval.list_gateway_approvals('gateway-owner')
    finally:
        approval.unregister_gateway_notify('gateway-owner')
        clear_session_vars(tokens)
