"""Out-of-tree consumer integration; explicitly selected app source fixture."""

import os
from pathlib import Path
import pytest
from hermes_cli.plugins import PluginManager


def test_standalone_plugin_refuses_native_execution(tmp_path, monkeypatch):
    import importlib.util

    spec = importlib.util.find_spec("hermes_worker_queue")
    if spec is None:
        pytest.skip(
            "install app in isolated test environment for cross-repository integration"
        )
    source = str(Path(spec.origin).parents[2])
    monkeypatch.syspath_prepend(str(Path(source) / "src"))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "plugins:\n  enabled: [hermes-worker-queue]\n"
    )
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "hermes-worker-queue").symlink_to(source, target_is_directory=True)
    manager = PluginManager()
    manager.discover_and_load()
    from agent.workload_registry import snapshot_registration, submit
    from agent.workload_provider import WorkloadSpec, WorkloadUnavailable

    assert snapshot_registration("hermes-worker-queue") is not None
    import argparse

    entry = manager._cli_commands.get("worker-queue")
    assert entry is not None, "supported CLI extension is missing"
    parser = argparse.ArgumentParser()
    entry["setup_fn"](parser)
    parsed = parser.parse_args(["--state", str(tmp_path), "status"])
    assert parsed.op == "status"
    with pytest.raises(WorkloadUnavailable):
        submit(
            {"mode": "required", "provider": "hermes-worker-queue"},
            WorkloadSpec("r", "terminal", "o", "d", 1),
        )
    manager.unload()
    assert snapshot_registration("hermes-worker-queue") is None
