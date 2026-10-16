"""Operator-approved legacy migration through real CLI parser and job store."""
import argparse


def test_cli_migrates_only_selected_job_without_scheduling(tmp_path, monkeypatch, capsys):
    from hermes_cli.subcommands.cron import build_cron_parser
    from hermes_cli.cron import cron_create, cron_edit
    from cron.jobs import create_job, get_job, list_jobs, use_cron_store

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(), cmd_cron=lambda args: None)
    with use_cron_store(tmp_path):
        legacy = create_job("", "every 2h", script="fixture.py", no_agent=True, paused=True)
        untouched = create_job("", "every 3h", script="other.py", no_agent=True, paused=True)
        before = get_job(legacy["id"])
        args = parser.parse_args(["cron", "edit", legacy["id"],
                                  "--estimated-peak-memory-mb", "64",
                                  "--estimated-retained-memory-mb", "32"])
        assert cron_edit(args) == 0
        after = get_job(legacy["id"])
        assert after["estimated_peak_memory_mb"] == 64
        assert after["estimated_retained_memory_mb"] == 32
        assert after["next_run_at"] == before["next_run_at"]
        assert after["enabled"] is False
        assert "estimated_peak_memory_mb" not in get_job(untouched["id"])
        scripts = tmp_path / "scripts"
        scripts.mkdir(exist_ok=True)
        (scripts / "new.py").write_text("print('fixture')\n")
        args = parser.parse_args(["cron", "create", "every 2h", "--script", "new.py",
                                  "--no-agent", "--paused", "--estimated-peak-memory-mb", "48"])
        assert cron_create(args) == 0
        assert any(j.get("estimated_peak_memory_mb") == 48 for j in list_jobs(include_disabled=True))
        assert "MiB" in capsys.readouterr().out
