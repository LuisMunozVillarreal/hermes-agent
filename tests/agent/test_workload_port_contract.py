"""Current-integration acceptance for the required workload provider API."""

from hermes_cli.plugins import PluginManager


def test_current_plugin_context_accepts_required_workload_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text("plugins:\n  enabled: [port-fixture]\n")
    plugin = tmp_path / "plugins" / "port-fixture"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(
        "name: port-fixture\nversion: 1.0.0\ndescription: execution owner fixture\n"
    )
    (plugin / "__init__.py").write_text(
        "from agent.workload_provider import WorkloadProvider\n"
        "class Provider(WorkloadProvider):\n"
        "    name = 'port-fixture'\n"
        "    def submit(self, spec): return {'ticket': spec.request_id}\n"
        "def register(ctx):\n"
        "    ctx.register_workload_provider(Provider())\n"
        "    from hermes_constants import get_hermes_home\n"
        "    (get_hermes_home() / 'registered').write_text('yes')\n"
    )
    manager = PluginManager()
    try:
        manager.discover_and_load()
        assert (tmp_path / "registered").exists(), "required execution owner did not load"
        from agent.workload_registry import submit
        from agent.workload_provider import WorkloadSpec
        spec = WorkloadSpec("request", "terminal", "owner", "domain", 1024)
        assert submit({"mode": "required", "provider": "port-fixture"}, spec)["ticket"] == "request"
    finally:
        manager.unload()
