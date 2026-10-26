"""Real paired provider: concurrent roots and nested work retain trusted ownership."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3

import pytest

from tests.agent.test_isolated_profile import profile_model as _fixture_profile_model, session_ids as _fixture_session_ids
profile_model = _fixture_profile_model
session_ids = _fixture_session_ids
from tests.agent.test_isolated_queue_integration import queue_home as _fixture_queue_home, wait_for as _fixture_wait_for, AGENT_CAP as _fixture_AGENT_CAP, TOOL_CAP as _fixture_TOOL_CAP
queue_home = _fixture_queue_home
wait_for = _fixture_wait_for
AGENT_CAP = _fixture_AGENT_CAP
TOOL_CAP = _fixture_TOOL_CAP


PLUGIN = '''
import json
import subprocess
from pathlib import Path
from hermes_constants import get_hermes_home
from agent.secret_scope import get_secret

from agent.workload_context import capture_commissioning
from agent.workload_boundary import popen
from agent.workload_provider import WorkloadUnavailable

def probe(args, **kwargs):
    rejected = []
    for field, value in args.items():
        try:
            popen('terminal', ['/bin/true'], **{field: value})
        except WorkloadUnavailable:
            rejected.append(field)
    child = popen('terminal', ['/bin/true'], stdin=subprocess.DEVNULL,
                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert child.wait(timeout=10) == 0
    return json.dumps({'home': str(get_hermes_home()),
        'secret': get_secret('PROFILE_TEST_SECRET'),
        'foreign': get_secret('FOREIGN_ONLY_SECRET'),

        'commissioning': capture_commissioning(), 'rejected': rejected})

def register(ctx):
    ctx.register_tool(name='profile_probe', toolset='profile_probe',
        schema={'name': 'profile_probe', 'description': 'Read test scope',
                'parameters': {'type': 'object', 'properties': {
                    'commissioning_context': {'type': 'object'},
                    'profile_id': {'type': 'string'}, 'session_id': {'type': 'string'}}}},
        handler=probe, check_fn=lambda: True)
'''


@pytest.mark.platforms("linux")
@pytest.mark.parametrize('same_profile', [False, True])
def test_concurrent_roots_nested_ownership_and_model_spoof_refusal(
        queue_home, profile_model, monkeypatch, same_profile):
    from agent.isolated_agent import IsolatedAgent
    from agent.secret_scope import (set_secret_scope, reset_secret_scope,
                                    is_multiplex_active, set_multiplex_active)
    from gateway.session_context import set_session_vars, clear_session_vars
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    from hermes_cli.plugins import get_plugin_manager

    base, state, client, _provider, backend = queue_home
    original_config = (base / 'config.yaml').read_text()
    homes = [base / 'profile-A', base / ('profile-A' if same_profile else 'profile-B')]
    for home in set(homes):
        home.mkdir()
        plugin = home / 'plugins' / 'profile-probe'
        plugin.mkdir(parents=True)
        (home / 'plugins' / 'hermes-worker-queue').symlink_to(
            (base / 'plugins' / 'hermes-worker-queue').resolve(), target_is_directory=True)
        (plugin / 'plugin.yaml').write_text('name: profile-probe\nversion: 1.0.0\ndescription: fixture\n')
        (plugin / '__init__.py').write_text(PLUGIN)
        (home / 'config.yaml').write_text(original_config.replace(
            '[hermes-worker-queue]', '[hermes-worker-queue, profile-probe]') +
            'tools:\n  tool_search: false\n')
    monkeypatch.setenv('FOREIGN_ONLY_SECRET', 'foreign-ambient-secret')
    monkeypatch.setenv('HERMES_SESSION_ID', 'foreign-ambient-session')
    url, requests = profile_model
    requests.arguments = json.dumps({'commissioning_context': {
        'profile_key': 'model-forged-profile', 'session_key': 'model-forged-session'},
        'profile_id': 'model-forged-profile', 'session_id': 'model-forged-session'})
    requests.release.clear()
    children, managers, expected = [], {}, []
    previous = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for index, home in enumerate(homes):
            token = set_hermes_home_override(str(home))
            secret = set_secret_scope({'PROFILE_TEST_SECRET': 'synthetic-' + home.name})
            routing = set_session_vars(session_id=f'commissioner-{index}', profile='forged-label')
            try:
                manager = get_plugin_manager()
                if home not in managers:
                    manager.discover_and_load()
                    managers[home] = manager
                children.append(IsolatedAgent(kind='delegation', model='fixture',
                    provider='custom', base_url=url, enabled_toolsets=['profile_probe'],
                    skip_memory=True, skip_context_files=True, quiet_mode=True, max_iterations=3,
                    workload_estimate={'peak_memory_bytes': AGENT_CAP // 2,
                        'retained_memory_bytes': AGENT_CAP // 2 - 2 * TOOL_CAP}))
                expected.append({'profile_key': str(home), 'session_key': f'commissioner-{index}'})
            finally:
                clear_session_vars(routing)
                reset_secret_scope(secret)
                reset_hermes_home_override(token)
        # Run in fresh contexts after both commissioning scopes have exited.
        with ThreadPoolExecutor(2) as pool:
            futures = [pool.submit(child.run_conversation, 'probe trusted ownership') for child in children]
            try:
                wait_for(lambda: len(requests) >= 2, timeout=35)
                roots = client.call('status')['jobs']
                assert len(roots) == 2 and all(row['state'] == 'running' for row in roots), roots
                requests.release.set()
                results = [future.result(timeout=40) for future in futures]
                assert all(result['completed'] for result in results), results
            finally:
                requests.release.set()
                for child in children:
                    child.close()
        wait_for(lambda: client.call('status')['reserved'] == 0)
        status = client.call('status')
        for index, (child, home) in enumerate(zip(children, homes)):
            ticket = child.process.workload_lease['ticket']
            root = next(row for row in status['jobs'] if row['ticket'] == ticket)
            observed = [json.loads(m['content']) for m in results[index]['messages']
                        if m['role'] == 'tool']
            assert observed and observed[0].get('commissioning') == expected[index], observed
            assert child.session_id != expected[index]['session_key']
            assert observed[0]['home'] == str(home)
            assert observed[0]['secret'] == 'synthetic-' + home.name
            assert observed[0]['foreign'] is None
            assert set(observed[0]['rejected']) == {'commissioning_context', 'profile_id', 'session_id'}
            assert child.session_id in session_ids(home)
            assert child.session_id not in session_ids(base)
            if not same_profile:
                assert child.session_id not in session_ids(homes[1 - index])
            profile_id = hashlib.sha256(('profile\0' + str(home)).encode()).hexdigest()
            session_id = hashlib.sha256(('session\0' + str(home) + '\0' +
                                         expected[index]['session_key']).encode()).hexdigest()
            descendants = [row for row in status['jobs'] if row['parent_ticket'] == ticket]
            assert descendants, status
            for row in [root, *descendants]:
                assert row['profile_id'] == profile_id and row['session_id'] == session_id, row
                assert row['state'] == 'finished' and backend.empty(row['unit']), row
        roots = [row for row in status['jobs'] if row['parent_ticket'] is None]
        assert (roots[0]['profile_id'] == roots[1]['profile_id']) == same_profile
        assert roots[0]['session_id'] != roots[1]['session_id']
        # Inspect both public status and durable storage, not just provider arguments.
        with sqlite3.connect(f'file:{state / "ledger.db"}?mode=ro', uri=True) as db:
            ledger = '\n'.join(db.iterdump())
        for raw in [str(home) for home in homes] + [item['session_key'] for item in expected] + [
                'model-forged', 'synthetic-', 'foreign-ambient']:
            assert raw not in json.dumps(status) and raw not in ledger
    finally:
        requests.release.set()
        for child in children:
            child.close()
        for home, manager in managers.items():
            token = set_hermes_home_override(str(home))
            try:
                manager.unload()
            finally:
                reset_hermes_home_override(token)
        set_multiplex_active(previous)
