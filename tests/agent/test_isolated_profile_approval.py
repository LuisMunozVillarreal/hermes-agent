"""Real queue approval parity under a foreign ambient profile, with child slack."""
from concurrent.futures import ThreadPoolExecutor
import json
import shlex

import pytest

from tests.agent.test_isolated_agent import local_model as _fixture_local_model
local_model = _fixture_local_model
from tests.agent.test_isolated_queue_integration import queue_home as _fixture_queue_home, wait_for as _fixture_wait_for, AGENT_CAP as _fixture_AGENT_CAP, TOOL_CAP as _fixture_TOOL_CAP
queue_home = _fixture_queue_home
wait_for = _fixture_wait_for
AGENT_CAP = _fixture_AGENT_CAP
TOOL_CAP = _fixture_TOOL_CAP


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('kind', ['delegation', 'cron-agent'])
@pytest.mark.parametrize('choice', ['deny', 'once'])
def test_commissioned_approval_parity(queue_home, local_model, monkeypatch, kind, choice):
    from agent.isolated_agent import IsolatedAgent
    from agent.secret_scope import (set_secret_scope, reset_secret_scope,
                                    set_multiplex_active, is_multiplex_active)
    from hermes_constants import (get_hermes_home, set_hermes_home_override,
                                  reset_hermes_home_override)
    from gateway.session_context import set_session_vars, clear_session_vars

    b, _state, client, _provider, backend = queue_home
    a = b / 'ambient-A'
    a.mkdir()
    mode = 'approve' if choice == 'once' else 'deny'
    opposite = 'deny' if choice == 'once' else 'approve'
    (a / 'config.yaml').write_text(f'approvals:\n  mode: manual\n  cron_mode: {opposite}\n')
    config = b / 'config.yaml'
    config.write_text(config.read_text() + f'approvals:\n  mode: manual\n  cron_mode: {mode}\n')
    monkeypatch.setenv('HERMES_HOME', str(a))
    monkeypatch.delenv('HERMES_SINGLE_QUERY_SESSION', raising=False)
    monkeypatch.delenv('HERMES_YOLO_MODE', raising=False)
    marker = b / 'approved-effect'
    # The deletion target is disposable and nonexistent; the marker proves actual
    # terminal execution rather than a model claiming that it obeyed policy.
    local_model[1].command = (f'rm -rf {shlex.quote(str(b / "nonexistent"))}; '
                              f'touch {shlex.quote(str(marker))}')
    decisions = []

    def approve(command, description, **options):
        decisions.append((str(get_hermes_home()), command))
        return choice

    home = set_hermes_home_override(str(b))
    secret = set_secret_scope({})
    previous = is_multiplex_active()
    set_multiplex_active(True)
    routing = set_session_vars(platform='', cron_session='')
    try:
        child = IsolatedAgent(kind=kind, approval_callback=approve,
            workload_estimate={'peak_memory_bytes': AGENT_CAP,
                'retained_memory_bytes': AGENT_CAP - 2 * TOOL_CAP},
            model='fixture', provider='custom', base_url=local_model[0],
            enabled_toolsets=['terminal'], max_iterations=3, skip_memory=True,
            skip_context_files=True, quiet_mode=True)
    finally:
        clear_session_vars(routing)
        reset_secret_scope(secret)
        reset_hermes_home_override(home)
        set_multiplex_active(previous)

    # Run after scope exit, as detached delegation does. A bounded future ensures
    # lack of nested slack cannot hang the file and cleanup cancels owned work.
    pool = ThreadPoolExecutor(1)
    future = pool.submit(child.run_conversation, 'execute the fixture command')
    try:
        result = future.result(timeout=40)
        assert result['completed'], result
        tools = [m for request in local_model[1] for m in request.get('messages', []) if m['role'] == 'tool']
        assert tools, result
        assert marker.exists() == (choice == 'once'), tools
        if choice == 'deny':
            assert any(json.loads(m['content']).get('status') == 'blocked' for m in tools), tools
        if kind == 'delegation':
            assert decisions and all(home == str(b) for home, _ in decisions), decisions
        else:
            assert not decisions, 'cron policy must resolve without an interactive callback'
        wait_for(lambda: client.call('status')['reserved'] == 0)
        status = client.call('status')
        roots = [r for r in status['jobs'] if r['parent_ticket'] is None]
        assert len(roots) == 1 and roots[0]['state'] == 'finished', status
        assert roots[0]['peak'] == AGENT_CAP
        assert roots[0]['retained'] == AGENT_CAP - 2 * TOOL_CAP
        descendants = [r for r in status['jobs'] if r['parent_ticket'] is not None]
        if choice == 'once':
            assert descendants, status
        assert all(r['parent_ticket'] == roots[0]['ticket'] for r in descendants), status
        assert all(backend.empty(r['unit']) for r in status['jobs'] if r.get('unit'))
        print(json.dumps({'kind': kind, 'choice': choice, 'marker': marker.exists(),
                          'reserved': status['reserved'], 'all_scopes_empty': True}))
    finally:
        child.close()
        pool.shutdown(wait=True, cancel_futures=True)
        monkeypatch.setenv('HERMES_HOME', str(b))
