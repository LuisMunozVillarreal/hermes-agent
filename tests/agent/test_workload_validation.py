"""Provider validation is actionable without exposing arbitrary provider errors."""

from types import SimpleNamespace

import pytest

from agent import workload_boundary, workload_provider


def install_provider(monkeypatch, popen):
    monkeypatch.setattr(
        workload_boundary, "workload_config", lambda: {"mode": "required"}
    )
    monkeypatch.setattr(
        workload_boundary,
        "select_provider",
        lambda config: SimpleNamespace(
            supported_kinds={"delegation"},
            popen=popen,
        ),
    )


def test_boundary_preserves_safe_numeric_validation(monkeypatch):
    error = workload_provider.WorkloadValidationError(
        "retained-exceeds-peak",
        peak_memory_bytes=256 * 1024**2,
        retained_memory_bytes=257 * 1024**2,
    )

    def reject(kind, args, *, config, **kwargs):
        raise error

    install_provider(monkeypatch, reject)
    with pytest.raises(workload_provider.WorkloadValidationError) as caught:
        workload_boundary.popen(
            "delegation", ["private-payload"], env={"TOKEN": "private-secret"}
        )
    assert caught.value is error
    message = str(caught.value)
    assert "retained_memory_bytes=269484032" in message
    assert "peak_memory_bytes=268435456" in message
    assert "must not exceed" in message
    assert "private" not in message


@pytest.mark.parametrize(
    "exception", [ValueError, RuntimeError, workload_provider.WorkloadUnavailable]
)
def test_legacy_provider_errors_remain_redacted(monkeypatch, exception):
    def reject(kind, args, *, config, **kwargs):
        assert "commissioning_context" not in kwargs
        assert "cancel_event" not in kwargs
        raise exception("private-payload private-secret")

    install_provider(monkeypatch, reject)
    with pytest.raises(
        workload_provider.WorkloadUnavailable,
        match="^workload execution-owner launch failed$",
    ):
        workload_boundary.popen("delegation", ["private-payload"])


def test_boundary_preserves_actionable_growth_minimum(monkeypatch):
    error = workload_provider.WorkloadValidationError(
        "growth-required",
        peak_memory_bytes=2448 * 1024**2,
        retained_memory_bytes=2048 * 1024**2,
    )

    def reject(kind, args, *, config, **kwargs):
        raise error

    install_provider(monkeypatch, reject)
    with pytest.raises(workload_provider.WorkloadValidationError) as caught:
        workload_boundary.popen("delegation", ["private-payload"])
    assert caught.value is error
    message = str(error)
    assert "growth required" in message
    assert "absolute" in message and "scheduler-approved" in message
    assert "peak_memory_bytes=2566914048" in message
    assert "retained_memory_bytes=2147483648" in message
    assert "private" not in message


def test_validation_diagnostics_never_interpolate_unvalidated_values():
    error = workload_provider.WorkloadValidationError(
        "private-secret",
        peak_memory_bytes="private-payload",
        retained_memory_bytes=True,
    )
    assert "private" not in str(error)
    assert "True" not in str(error)
    assert "workload" in str(error).lower()
