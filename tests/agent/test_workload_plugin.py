"""Fresh real plugin discovery, profile separation and unload."""

import pytest
from hermes_cli.plugins import PluginManager


def test_real_loader_registers_and_unloads_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("plugins:\n  enabled: [fixture]\n")
    plugin = tmp_path / "plugins" / "fixture"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: fixture\nversion: 1.0.0\ndescription: workload fixture\n"
    )
    (plugin / "__init__.py").write_text(
        "from agent.workload_provider import WorkloadProvider\n"
        "class Fixture(WorkloadProvider):\n"
        "    name = 'fixture'\n"
        "    def submit(self, spec): return {'state': 'queued', 'ticket': spec.request_id}\n"
        "def register(ctx): ctx.register_workload_provider(Fixture())\n"
    )
    manager = PluginManager()
    manager.discover_and_load()
    from agent import workload_registry as r
    from agent.workload_provider import WorkloadSpec, WorkloadUnavailable

    assert hasattr(r, "submit"), "real discovery has no workload registry"
    cfg = {"mode": "required", "provider": "fixture"}
    spec = WorkloadSpec("r", "terminal", "owner", "domain", 1024)
    assert r.submit(cfg, spec)["ticket"] == "r"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "other"))
    with pytest.raises(WorkloadUnavailable):
        r.submit(cfg, spec)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    manager.unload()
    with pytest.raises(WorkloadUnavailable):
        r.submit(cfg, spec)
