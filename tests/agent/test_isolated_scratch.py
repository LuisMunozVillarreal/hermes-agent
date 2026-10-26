"""The worker control socket respects the caller's private scratch directory."""
from pathlib import Path
import tempfile

import pytest


def test_control_socket_uses_configured_temporary_directory(tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HERMES_HOME', str(home))
    scratch = tmp_path.parent / 's'
    scratch.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(scratch))
    seen = []

    def stop_before_launch(argv, **kwargs):
        seen.append(Path(argv[-1]))
        raise RuntimeError('fixture stops before worker launch')

    from agent.isolated_agent import IsolatedAgent
    child = IsolatedAgent(kind='delegation', launcher=stop_before_launch,
                          model='fixture', provider='custom',
                          base_url='http://127.0.0.1:1/v1')
    try:
        with pytest.raises(RuntimeError, match='fixture stops'):
            child.run_conversation('no model call')
        assert seen[0].parent.parent == scratch
        assert not seen[0].parent.exists()
    finally:
        child.close()
