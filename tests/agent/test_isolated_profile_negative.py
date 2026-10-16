"""Reject missing commissioning authority before config/plugins or agents run."""
import subprocess

import pytest

from tests.agent.test_isolated_agent import isolated_home as _fixture_isolated_home, local_model as _fixture_local_model
isolated_home = _fixture_isolated_home
local_model = _fixture_local_model


def test_multiplex_without_secret_scope_refuses_commission(isolated_home):
    from agent.isolated_agent import IsolatedAgent
    from agent.secret_scope import (UnscopedSecretError, is_multiplex_active,
        set_multiplex_active, set_secret_scope, reset_secret_scope)

    previous = is_multiplex_active()
    token = set_secret_scope(None)
    set_multiplex_active(True)
    try:
        with pytest.raises(UnscopedSecretError, match='requires a profile secret scope'):
            IsolatedAgent(kind='delegation', launcher=subprocess.Popen)
        assert not (isolated_home / 'state.db').exists()
    finally:
        set_multiplex_active(previous)
        reset_secret_scope(token)


def test_replaced_profile_refuses_before_launch(isolated_home):
    from agent.isolated_agent import IsolatedAgent

    launched = []
    child = IsolatedAgent(kind='delegation', launcher=lambda *a, **k: launched.append(a))
    isolated_home.rename(isolated_home.with_name(isolated_home.name + '-old'))
    isolated_home.mkdir()
    try:
        with pytest.raises(ValueError, match='identity changed'):
            child.run_conversation('must not launch')
        assert launched == []
        assert not (isolated_home / 'state.db').exists()
    finally:
        child.close()


@pytest.mark.parametrize('fault', ['launch_home', 'identity', 'missing_secrets'])
def test_worker_refuses_invalid_profile_before_agent_import(isolated_home, local_model, fault):
    from agent.isolated_agent import IsolatedAgent

    other = isolated_home / 'other'
    other.mkdir()
    # If runtime plugin discovery happens before validation, leave evidence.
    plugin = isolated_home / 'plugins' / 'must-not-import'
    plugin.mkdir(parents=True)
    (plugin / 'plugin.yaml').write_text('name: must-not-import\nversion: 1.0.0\ndescription: fixture\n')
    marker = isolated_home / 'imported'
    (plugin / '__init__.py').write_text(
        f'from pathlib import Path\nPath({str(marker)!r}).touch()\ndef register(ctx): pass\n')
    (isolated_home / 'config.yaml').write_text('plugins:\n  enabled: [must-not-import]\n')

    def launch(args, **options):
        # Mutate only after the parent validated the launch environment. The real
        # authenticated worker must independently reject the resulting context.
        if fault == 'launch_home':
            options['env']['HERMES_HOME'] = str(other)
        elif fault == 'identity':
            child._profile_context['identity'][1] += 1
        else:
            child._profile_context.update(multiplex=True, secrets=None)
        return subprocess.Popen(args, **options)

    child = IsolatedAgent(kind='delegation', launcher=launch,
        model='fixture', provider='custom', base_url=local_model[0],
        enabled_toolsets=[], skip_memory=True, skip_context_files=True, quiet_mode=True)
    expected = {'launch_home': 'profile mismatch', 'identity': 'identity changed',
                'missing_secrets': 'missing isolated profile secret scope'}[fault]
    try:
        with pytest.raises(RuntimeError, match=expected):
            child.run_conversation('must not reach model')
        assert not local_model[1]
        assert not marker.exists()
        assert not (isolated_home / 'state.db').exists()
        assert not (other / 'state.db').exists()
        assert child.process.poll() is not None
    finally:
        child.close()
