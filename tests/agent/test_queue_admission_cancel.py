"""Queued interruption through the real discovered app and Unix coordinator."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from tests.agent.test_isolated_queue_integration import BUDGET as _fixture_BUDGET, gated_model as _fixture_gated_model, queue_home as _fixture_queue_home, wait_for as _fixture_wait_for
BUDGET = _fixture_BUDGET
gated_model = _fixture_gated_model
queue_home = _fixture_queue_home
wait_for = _fixture_wait_for


@pytest.mark.parametrize("kind", ["delegation", "cron-agent"])
def test_interrupt_cancels_queued_admission_before_capacity_release(
    queue_home, gated_model, kind
):
    from agent.isolated_agent import IsolatedAgent

    home, _state, client, _provider, backend = queue_home
    url, _settings, entered, _release, requests = gated_model
    child = IsolatedAgent(
        kind=kind,
        model="fixture",
        provider="custom",
        base_url=url,
        enabled_toolsets=[],
        skip_memory=True,
        skip_context_files=True,
        quiet_mode=True,
    )
    blocker = client.call(
        "submit",
        request="cancel-blocker",
        peak=BUDGET,
        ttl=60,
        cwd=str(home),
        argv=["/bin/sleep", "60"],
    )
    try:
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(child.run_conversation, "must never reach model")
            try:
                queued = wait_for(
                    lambda: next(
                        (
                            r
                            for r in client.call("status")["jobs"]
                            if r["ticket"] != blocker["ticket"]
                            and r["state"] == "queued"
                        ),
                        None,
                    )
                )
                assert child.interrupt("cancel pending admission")
                result = future.result(timeout=3)
                assert result["interrupted"] is True
                assert result["completed"] is False
                status = client.call("status")
                row = next(r for r in status["jobs"] if r["ticket"] == queued["ticket"])
                assert row["state"] == "cancelled"
                assert row["unit"] is None
                assert status["reserved"] == BUDGET
                assert not backend.empty(blocker["unit"])
                assert child.process is None
                assert not entered.is_set() and not requests
            finally:
                # Release only after assertions; also unblock the unfixed implementation.
                client.call("cancel", ticket=blocker["ticket"])
                future.result(timeout=20)
        wait_for(lambda: client.call("status")["reserved"] == 0)
        assert all(
            backend.empty(r["unit"])
            for r in client.call("status")["jobs"]
            if r.get("unit")
        )
        assert not entered.is_set() and not requests
    finally:
        child.close()
