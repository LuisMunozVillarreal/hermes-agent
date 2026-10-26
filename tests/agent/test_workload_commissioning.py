"""Trusted reporting context at the generic execution-owner boundary."""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest


def test_concurrent_launches_use_scoped_identity_not_labels(tmp_path, monkeypatch):
    from agent import workload_boundary as boundary
    from gateway.session_context import set_session_vars, clear_session_vars
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override

    seen = {}
    barrier = Barrier(2)

    class Provider:
        supported_kinds = {'terminal'}

        def popen(self, kind, args, *, config, commissioning_context=None, **kwargs):
            seen[args[0]] = commissioning_context
            return args[0]

    monkeypatch.setattr(boundary, 'select_provider', lambda config: Provider())
    monkeypatch.setattr(boundary, 'workload_config', lambda: {
        'profile_id': 'forged-config', 'session_id': 'forged-config'})
    monkeypatch.setenv('HERMES_SESSION_ID', 'foreign-ambient-session')
    monkeypatch.setenv('HERMES_PROFILE', 'foreign-ambient-label')

    def launch(name):
        home = tmp_path / name
        home.mkdir()
        token = set_hermes_home_override(str(home))
        routing = set_session_vars(session_id='session-' + name, profile='forged-label')
        try:
            barrier.wait(timeout=5)
            return boundary.popen('terminal', [name])
        finally:
            clear_session_vars(routing)
            reset_hermes_home_override(token)

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(launch, ['A', 'B']))
    assert results == ['A', 'B']
    assert seen == {name: {'profile_key': str(tmp_path / name),
                          'session_key': 'session-' + name} for name in results}

    # An unbound task must not borrow ambient session identity from another task.
    contextvars.Context().run(boundary.popen, 'terminal', ['unbound'])
    assert seen['unbound']['session_key'] is None


@pytest.mark.parametrize('field', ['commissioning_context', 'profile_id', 'session_id'])
def test_boundary_refuses_caller_ownership(field, monkeypatch):
    from agent import workload_boundary as boundary
    from agent.workload_provider import WorkloadUnavailable

    calls = []
    class Provider:
        supported_kinds = {'terminal'}
        def popen(self, kind, args, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(boundary, 'select_provider', lambda config: Provider())
    monkeypatch.setattr(boundary, 'workload_config', dict)
    with pytest.raises(WorkloadUnavailable, match='commissioning'):
        boundary.popen('terminal', ['must-not-launch'], **{field: 'forged'})
    assert calls == []


def test_legacy_provider_and_disabled_launch_receive_no_metadata(monkeypatch):
    from agent import workload_boundary as boundary
    calls = []
    class Provider:
        supported_kinds = {'terminal'}
        def popen(self, kind, args, *, config, **kwargs):
            calls.append(kwargs)
    monkeypatch.setattr(boundary, 'workload_config', dict)
    monkeypatch.setattr(boundary, 'select_provider', lambda config: Provider())
    boundary.popen('terminal', ['legacy'])
    monkeypatch.setattr(boundary, 'select_provider', lambda config: None)
    monkeypatch.setattr(boundary.subprocess, 'Popen', lambda args, **kwargs: calls.append(kwargs))
    boundary.popen('terminal', ['disabled'])
    assert calls == [{}, {}]


@pytest.mark.parametrize('ownership', [None, {'profile_key': '/root-profile', 'session_key': 'root-session'}])
def test_bootstrap_snapshot_is_thread_safe_and_legacy_stays_null(monkeypatch, ownership):
    from agent import workload_context as context
    from gateway.session_context import scoped_current_session_id

    monkeypatch.setattr(context, '_worker_context', None)
    monkeypatch.setattr(context, '_worker_commissioning', None)
    expected = None if ownership is None else dict(ownership)
    context.install_worker_context({'ticket': 'a' * 32, 'generation': 'b' * 32}, ownership)
    if ownership is not None:
        ownership['session_key'] = 'mutated-input'
    with scoped_current_session_id('new-child-session'):
        assert context.capture_commissioning() == expected
        with ThreadPoolExecutor(1) as pool:
            assert pool.submit(context.capture_commissioning).result(timeout=5) == expected
    result = context.capture_commissioning()
    if result is not None:
        result['profile_key'] = 'mutated-return'
    assert context.capture_commissioning() == expected
    with pytest.raises(RuntimeError, match='already installed'):
        context.install_worker_context({'ticket': 'c' * 32, 'generation': 'd' * 32})


@pytest.mark.parametrize('context', [
    {}, {'profile_key': 'p'}, {'profile_key': 'p', 'session_key': None, 'extra': True},
    {'profile_key': '', 'session_key': None}, {'profile_key': 'x' * 4097, 'session_key': None},
    {'profile_key': 'p', 'session_key': False}, {'profile_key': 'p', 'session_key': 'bad\0id'},
])
def test_invalid_bootstrap_ownership_does_not_install_worker(monkeypatch, context):
    from agent import workload_context
    monkeypatch.setattr(workload_context, '_worker_context', None)
    with pytest.raises(ValueError, match='commissioning'):
        workload_context.install_worker_context({'ticket': 'a' * 32, 'generation': 'b' * 32}, context)
    assert workload_context.worker_lease() is None


def test_config_and_agent_kwargs_cannot_install_ownership(tmp_path, monkeypatch):
    from agent.workload_boundary import workload_config
    from agent.isolated_protocol import AgentConstruction

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('HERMES_IGNORE_USER_CONFIG', raising=False)
    (tmp_path / 'config.yaml').write_text(
        'workloads:\n  mode: disabled\n  commissioning_context: forged\n'
        '  profile_id: forged\n  session_id: forged\n')
    assert not {'commissioning_context', 'profile_id', 'session_id'} & workload_config().keys()
    for field in ('commissioning_context', 'profile_id', '_commissioning_context', 'profile_context'):
        with pytest.raises(ValueError):
            AgentConstruction('delegation', {field: 'forged'})
