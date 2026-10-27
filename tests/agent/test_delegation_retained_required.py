import pytest

from agent import workload_estimates as estimates
from agent.workload_provider import WorkloadValidationError


def test_delegation_requires_retained(monkeypatch):
    monkeypatch.setattr(estimates, "model_peak_required", lambda kind: True)
    with pytest.raises(WorkloadValidationError, match="estimated_retained_memory_mb"):
        estimates.validate_model_estimates("delegation", 1024)
    assert estimates.RETAINED_FIELD in estimates.model_required_fields(
        "delegation", ["goal"]
    )


def test_delegation_requires_child_capacity(monkeypatch):
    monkeypatch.setattr(estimates, "model_peak_required", lambda kind: True)
    with pytest.raises(WorkloadValidationError, match="child"):
        estimates.validate_model_estimates("delegation", 1024, 1024)
    estimates.validate_model_estimates("delegation", 3072, 1024)


def test_leaf_and_unmanaged_compatibility(monkeypatch):
    monkeypatch.setattr(estimates, "model_peak_required", lambda kind: True)
    estimates.validate_model_estimates("terminal", 128)
    assert estimates.RETAINED_FIELD not in estimates.model_required_fields(
        "terminal", ["command"]
    )
    monkeypatch.setattr(estimates, "model_peak_required", lambda kind: False)
    estimates.validate_model_estimates("delegation", 1024)
    assert estimates.RETAINED_FIELD not in estimates.model_required_fields(
        "delegation", ["goal"]
    )
