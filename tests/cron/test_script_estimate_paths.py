"""Exercise native script paths without the shared systemd acceptance fixture."""
from concurrent.futures import ThreadPoolExecutor
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def script_provider(tmp_path, monkeypatch):
    from agent.workload_registry import register_provider, restore_registration

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        "workloads:\n  mode: required\n  provider: script-estimates\n")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "probe.py").write_text("print('estimate script output')")
    launches = []

    def popen(kind, args, *, config, **kwargs):
        launches.append((kind, {k: kwargs.pop(k) for k in
                              ("peak_memory_bytes", "retained_memory_bytes") if k in kwargs}))
        return subprocess.Popen(args, **kwargs)

    provider = SimpleNamespace(name="script-estimates", api_version=1,
                               supported_kinds=frozenset({"cron-script", "cron-agent"}), popen=popen)
    register_provider(provider)
    try:
        yield tmp_path, launches
    finally:
        restore_registration(provider.name, provider, None)


def run_path(path, job):
    from cron.monitor import _run_monitor_source
    from cron.scheduler import _run_no_agent_job, run_job
    from cron.scheduler_prompt import _build_job_prompt
    from cron.scheduler_script import _run_job_script_with_claim_heartbeat

    if path == "monitor":
        return _run_monitor_source({**job, "monitor_script": "probe.py"})
    if path == "heartbeat":
        return _run_job_script_with_claim_heartbeat(job, "probe.py")
    if path == "inline":
        return True, _build_job_prompt(job)
    if path == "no_agent":
        result = _run_no_agent_job(job, job["id"], job["id"], None)
    else:
        result = run_job(job)
    return result[0], result[2] or result[3]


@pytest.mark.parametrize("path", ["monitor", "heartbeat", "inline", "no_agent", "run_job"])
def test_script_paths_use_job_estimates_and_restore_context(script_provider, path):
    from agent.workload_estimates import current_estimate, estimate_scope

    _home, launches = script_provider
    job = {"id": "estimate", "prompt": "Report output", "script": "probe.py", "no_agent": True,
           "estimated_peak_memory_mb": 48, "estimated_retained_memory_mb": 24}
    with estimate_scope({"cron-script"}, {"peak_memory_bytes": 123}):
        ok, output = run_path(path, job)
        assert ok and "estimate script output" in output, output
        assert launches[-1] == ("cron-script", {"peak_memory_bytes": 48 * 1024**2,
                                               "retained_memory_bytes": 24 * 1024**2})
        assert current_estimate("cron-script") == {"peak_memory_bytes": 123}
        legacy = {k: v for k, v in job.items() if not k.startswith("estimated_")}
        assert run_path(path, legacy)[0]
        assert launches[-1] == ("cron-script", {})
    assert current_estimate("cron-script") == {}


@pytest.mark.parametrize("path", ["monitor", "heartbeat", "no_agent", "run_job"])
def test_invalid_job_estimate_never_launches_script(script_provider, path):
    _home, launches = script_provider
    ok, output = run_path(path, {"id": "invalid", "script": "probe.py", "no_agent": True,
                                 "estimated_peak_memory_mb": True})
    assert not ok and "positive integer MiB" in output
    assert launches == []


def test_concurrent_monitors_keep_job_estimates(script_provider):
    _home, launches = script_provider
    def run(peak):
        return run_path("monitor", {"id": str(peak), "estimated_peak_memory_mb": peak})
    with ThreadPoolExecutor(2) as pool:
        assert all(ok for ok, _ in pool.map(run, [40, 48]))
    assert sorted(memory["peak_memory_bytes"] for _, memory in launches) == [40 * 1024**2, 48 * 1024**2]
