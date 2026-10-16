"""Cron estimates are durable job data, not a reservation at creation time."""
import json

import pytest


def test_estimates_create_update_reload_and_run(tmp_path, monkeypatch):
    from cron.jobs import create_job, get_job, update_job, use_cron_store
    from tools.cronjob_tools import _cronjob_handler
    from cron.scheduler import run_job

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with use_cron_store(tmp_path):
        job = create_job("", "every 2h", script="test.py", no_agent=True,
                         estimated_peak_memory_mb=48)
        assert get_job(job["id"])["estimated_peak_memory_mb"] == 48
        result = json.loads(_cronjob_handler({"action": "update", "job_id": job["id"],
                                             "estimated_peak_memory_mb": 56}))
        assert result["success"], result
        assert result["job"]["estimated_peak_memory_mb"] == 56
        assert get_job(job["id"])["estimated_peak_memory_mb"] == 56
        with pytest.raises(ValueError):
            update_job(job["id"], {"estimated_peak_memory_mb": True})
        assert get_job(job["id"])["estimated_peak_memory_mb"] == 56
        # Raw/legacy records are validated again at run time, before script launch.
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        marker = tmp_path / "launched"
        (scripts / "test.py").write_text(f"from pathlib import Path; Path({str(marker)!r}).touch()")
        invalid = {**get_job(job["id"]), "estimated_peak_memory_mb": 0}
        result = run_job(invalid)
        assert not result[0] and "memory" in result[3], result
        assert not marker.exists()


@pytest.mark.parametrize("updates", [
    {"estimated_peak_memory_mb": 0}, {"estimated_peak_memory_mb": 2**43},
    {"estimated_peak_memory_mb": 1.5}, {"estimated_retained_memory_mb": 65},
    {"estimated_peak_memory_mb": None},
])
def test_invalid_merged_estimates_do_not_change_storage(tmp_path, updates):
    from cron.jobs import create_job, get_job, update_job, use_cron_store
    with use_cron_store(tmp_path):
        job = create_job("", "every 2h", script="test.py", no_agent=True,
                         estimated_peak_memory_mb=64, estimated_retained_memory_mb=32)
        before = get_job(job["id"])
        with pytest.raises(ValueError):
            update_job(job["id"], updates)
        assert get_job(job["id"]) == before
