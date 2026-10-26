"""Generic execution-owner selection; no scheduler policy or inline fallback."""

import subprocess
from agent.workload_provider import WorkloadCancelled, WorkloadUnavailable
from agent.workload_registry import select_provider


def workload_config():
    from hermes_cli.config_effective import load_user_config_effective

    try:
        config = load_user_config_effective(fail_closed=True).get("workloads", {})
    except Exception as exc:
        raise WorkloadUnavailable("workload configuration unavailable") from exc
    if not isinstance(config, dict):
        raise WorkloadUnavailable("invalid workload configuration")
    config = dict(config)
    for key in ("parent_lease", "commissioning_context", "profile_id", "session_id"):
        config.pop(key, None)
    from agent.workload_context import worker_lease
    lease = worker_lease()
    if lease is not None:
        config["parent_lease"] = lease
    return config


def request_growth(peak_memory_bytes, *, retained_memory_bytes=None):
    """Request absolute targets for this authenticated worker, never a caller lease."""
    from agent.workload_provider import positive_bytes
    from agent.workload_context import worker_lease
    positive_bytes(peak_memory_bytes)
    if retained_memory_bytes is not None:
        if positive_bytes(retained_memory_bytes) > peak_memory_bytes:
            raise ValueError("retained memory exceeds subtree peak")
    lease = worker_lease()
    config = workload_config()
    provider = select_provider(config)
    if lease is None or provider is None:
        raise WorkloadUnavailable("growth requires an admitted worker")
    return provider.request_growth(lease, peak_memory_bytes,
        retained_memory_bytes=retained_memory_bytes, config=config)


def require_unmanaged_allowed(kind):
    if workload_config().get("mode", "disabled") != "disabled":
        raise WorkloadUnavailable(
            f"Required workload protection: unsupported execution adapter: {kind}"
        )


def execution_provider(kind):
    provider = select_provider(workload_config())
    if provider is not None and kind not in provider.supported_kinds:
        raise WorkloadUnavailable(
            f"Required workload protection: unsupported execution adapter: {kind}"
        )
    return provider


def popen(kind, args, *, cancel_event=None, **kwargs):
    if set(kwargs) & {"commissioning_context", "profile_id", "session_id"}:
        raise WorkloadUnavailable("ownership requires trusted commissioning context")
    config = workload_config()
    provider = select_provider(config)
    if provider is None:
        return subprocess.Popen(args, **kwargs)
    if cancel_event is None and kind in {"terminal", "terminal-pty"}:
        from agent.workload_cancellation import terminal_cancellation
        cancel_event = terminal_cancellation()
    if cancel_event is not None and cancel_event.is_set():
        raise WorkloadCancelled("workload launch cancelled")
    if kind not in provider.supported_kinds:
        raise WorkloadUnavailable(f"unsupported workload execution: {kind}")
    try:
        # Only an explicit parameter opts in: legacy **kwargs providers forward
        # straight to subprocess.Popen, which cannot accept a cancellation token.
        import inspect
        parameters = inspect.signature(provider.popen).parameters
        if cancel_event is not None and "cancel_event" in parameters:
            kwargs["cancel_event"] = cancel_event
        ownership = parameters.get("commissioning_context")
        if ownership is not None and ownership.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            from agent.workload_context import capture_commissioning
            kwargs["commissioning_context"] = capture_commissioning()
        from agent.workload_estimates import current_estimate
        return provider.popen(kind, args, config=config, **current_estimate(kind), **kwargs)
    except InterruptedError as exc:
        raise WorkloadCancelled("workload launch cancelled") from exc
    except Exception as exc:
        raise WorkloadUnavailable("workload execution-owner launch failed") from exc
