"""Per-request estimates survive native launch, never shell environment state."""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.agent.test_workload_execution import managed_home as _fixture_managed_home
managed_home = _fixture_managed_home


@pytest.mark.parametrize("mode", ["foreground", "background", "pty", "promoted"])
def test_terminal_estimate_reaches_real_ledger(managed_home, mode):
    from tools.terminal_tool import _handle_terminal
    from hermes_worker_queue.client import Client

    home, state = managed_home
    from tools.process_registry import process_registry
    options = {"foreground": {}, "background": {"background": True},
               "pty": {"background": True, "pty": True}, "promoted": {"timeout": 700}}[mode]
    result = json.loads(_handle_terminal({
        "command": "printf estimate-native", "workdir": str(home),
        "estimated_peak_memory_mb": 48, **options,
    }, task_id="estimate-native-" + mode))
    if result.get("session_id"):
        session = process_registry.get(result["session_id"])
        session._reader_thread.join(timeout=8)
        assert session.exited
    assert not result.get("error"), result
    rows = Client(state).call("status")["jobs"]
    assert rows and all(row["peak"] == 48 * 1024**2 for row in rows), rows
    assert all(row["estimate_source"] == "job" for row in rows), rows


@pytest.mark.parametrize("value", [True, False, 0, -1, 1.5, 1.0, "64", float("inf"), 2**43])
def test_terminal_invalid_estimate_never_executes(tmp_path, value):
    from tools.terminal_tool import _handle_terminal

    marker = tmp_path / "must-not-exist"
    result = json.loads(_handle_terminal({"command": f"touch {marker}",
                                        "estimated_peak_memory_mb": value}))
    assert result.get("error"), result
    assert not marker.exists()


def test_concurrent_estimates_do_not_leak_or_become_environment(managed_home):
    from tools.terminal_tool import _handle_terminal
    from agent.workload_estimates import current_estimate
    from hermes_worker_queue.client import Client
    home, state = managed_home
    def run(mib):
        result = json.loads(_handle_terminal({
            "command": "env", "workdir": str(home), "estimated_peak_memory_mb": mib,
        }, task_id=f"concurrent-estimate-{mib}"))
        assert not result.get("error"), result
        assert "peak_memory" not in result["output"]
        assert current_estimate("terminal") == {}
    with ThreadPoolExecutor(2) as pool:
        list(pool.map(run, [40, 48]))
    rows = Client(state).call("status")["jobs"]
    assert {r["peak"] for r in rows} == {40 * 1024**2, 48 * 1024**2}
    assert all(r["estimate_source"] == "job" for r in rows)
    run_default = json.loads(_handle_terminal({"command": "printf default", "workdir": str(home)},
                                             task_id="default-after-concurrent"))
    assert not run_default.get("error"), run_default
    rows = Client(state).call("status")["jobs"]
    assert any(r["peak"] == 64 * 1024**2 and r["estimate_source"] == "operator" for r in rows)


def test_cron_script_persisted_estimate_and_legacy_default(managed_home):
    from cron.jobs import create_job, get_job, use_cron_store
    from cron.scheduler import run_job
    from hermes_worker_queue.client import Client
    home, state = managed_home
    scripts = home / "scripts"
    scripts.mkdir()
    (scripts / "small.py").write_text("print('stored-estimate')")
    with use_cron_store(home):
        job = create_job("", "every 2h", script="small.py", no_agent=True,
                         estimated_peak_memory_mb=48)
        assert Client(state).call("status")["jobs"] == []
        assert run_job(get_job(job["id"]))[0]
        rows = Client(state).call("status")["jobs"]
        assert len(rows) == 1 and rows[0]["peak"] == 48 * 1024**2
        assert rows[0]["estimate_source"] == "job"
        legacy = create_job("", "every 2h", script="small.py", no_agent=True)
        assert "estimated_peak_memory_mb" not in get_job(legacy["id"])
        assert run_job(legacy)[0]
        assert any(r["estimate_source"] == "operator" for r in Client(state).call("status")["jobs"])


def test_required_missing_estimate_refuses_legacy_script(managed_home):
    from cron.scheduler import run_job
    from hermes_worker_queue.client import Client
    import hermes_yaml as yaml
    home, state = managed_home
    config = yaml.safe_load((home / "config.yaml").read_text())
    del config["workloads"]["peak_memory_bytes"]
    (home / "config.yaml").write_text(yaml.safe_dump(config))
    scripts = home / "scripts"
    scripts.mkdir()
    marker = home / "never"
    (scripts / "small.py").write_text(f"from pathlib import Path; Path({str(marker)!r}).touch()")
    result = run_job({"id": "legacy", "script": "small.py", "no_agent": True})
    assert not result[0], result
    assert not marker.exists()
    assert Client(state).call("status")["jobs"] == []


def test_monitor_persisted_estimate_reaches_real_ledger(managed_home):
    from cron.jobs import create_job, get_job, use_cron_store
    from cron.monitor import _run_monitor_source
    from hermes_worker_queue.client import Client

    home, state = managed_home
    scripts = home / "scripts"
    scripts.mkdir()
    (scripts / "monitor.py").write_text("print('monitor-estimate')")
    with use_cron_store(home):
        job = create_job("Report changes", "every 2h", monitor_script="monitor.py",
                         estimated_peak_memory_mb=48)
        assert Client(state).call("status")["jobs"] == []
        ok, output = _run_monitor_source(get_job(job["id"]))
        assert ok and output == "monitor-estimate", output
        rows = Client(state).call("status")["jobs"]
        assert len(rows) == 1 and rows[0]["kind"] == "cron-script", rows
        assert rows[0]["peak"] == 48 * 1024**2 and rows[0]["estimate_source"] == "job"
        assert Client(state).call("status")["reserved"] == 0


def test_static_schemas_agree_with_integer_validation():
    from copy import deepcopy
    import jsonschema
    from tools.terminal_tool import TERMINAL_SCHEMA
    from tools.delegate_tool import DELEGATE_TASK_SCHEMA, _build_dynamic_schema_overrides
    from tools.cronjob_tools import CRONJOB_SCHEMA, _cronjob_schema_overrides
    from agent.workload_estimates import MAX_MEMORY_MIB, estimate_bytes
    schemas = [TERMINAL_SCHEMA["parameters"]["properties"],
               DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"],
               CRONJOB_SCHEMA["parameters"]["properties"]]
    before = deepcopy(schemas)
    for properties in schemas:
        assert not {"parent_lease", "generation", "peak_memory_bytes"}.intersection(properties)
        schema = properties["estimated_peak_memory_mb"]
        jsonschema.validate(MAX_MEMORY_MIB, schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(MAX_MEMORY_MIB + 1, schema)
        assert estimate_bytes(MAX_MEMORY_MIB)["peak_memory_bytes"] < 2**63
    _build_dynamic_schema_overrides()
    _cronjob_schema_overrides()
    assert schemas == before
