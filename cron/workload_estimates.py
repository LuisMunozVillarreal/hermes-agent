"""Keep one cron fire's estimates local through scripts and agent construction."""
from functools import wraps

from agent.workload_estimates import PEAK_FIELD, RETAINED_FIELD, estimate_bytes, estimate_scope


def with_job_estimates(run):
    @wraps(run)
    def scoped(job, **kwargs):
        try:
            metadata = estimate_bytes(job.get(PEAK_FIELD), job.get(RETAINED_FIELD))
        except ValueError as exc:
            return False, "", "", str(exc)
        with estimate_scope({"cron-script", "cron-agent"}, metadata):
            return run(job, **kwargs)
    return scoped


def with_job_script_estimates(run):
    """Cover script entry points also called outside ``scheduler.run_job``."""
    @wraps(run)
    def scoped(job, *args, **kwargs):
        try:
            metadata = estimate_bytes(job.get(PEAK_FIELD), job.get(RETAINED_FIELD))
        except ValueError as exc:
            return False, str(exc)
        with estimate_scope({"cron-script"}, metadata):
            return run(job, *args, **kwargs)
    return scoped
