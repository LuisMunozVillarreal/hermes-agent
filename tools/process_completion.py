"""Optional completion provenance supplied by process providers.

Providers keep a nonzero integer returncode for Popen compatibility when the
transport completes without a child status. They set returncode_known=False in
process_completion and supply bounded, sanitized diagnostics. Core carries that
receipt, without interpreting provider policy or treating the integer as an exit.
"""

from copy import deepcopy


def completion_fields(process) -> dict:
    completion = getattr(process, "process_completion", None)
    if isinstance(completion, dict) and type(completion.get("returncode_known")) is bool:
        return {"process_completion": deepcopy(completion)}
    return {}


def exit_status_unknown(result: dict) -> bool:
    completion = result.get("process_completion")
    return isinstance(completion, dict) and completion.get("returncode_known") is False


def reported_exit_code(process, fallback):
    """For provider receipts, prefer observation to a caller's signal guess."""
    fields = completion_fields(process)
    if not fields:
        return fallback
    if exit_status_unknown(fields):
        return None
    code = getattr(process, "returncode", None)
    if code is None:
        code = getattr(process, "exitstatus", None)
    return code if type(code) is int else fallback
