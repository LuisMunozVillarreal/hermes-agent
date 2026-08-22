"""Shared helpers for launching Hermes subprocesses in user systemd scopes.

This module centralizes the scoped-launch policy used by terminal execution
paths. Both foreground and background command launchers should call through
these helpers so user-visible behavior is identical across execution modes.
"""

import logging
import os
import platform
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional

_IS_WINDOWS = platform.system() == "Windows"
logger = logging.getLogger(__name__)

_SYSTEMD_SCOPE_AVAILABLE: Optional[bool] = None
_SYSTEMD_SCOPE_PROBE_LOCK = threading.Lock()
_SYSTEMD_SCOPE_PROBED_AT = 0.0
_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS = 60.0
_MIN_WORKER_MEMORY_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_WORKER_MEMORY_MAX_BYTES = 1024 * 1024 * 1024
_WORKER_MEMORY_MAX_CAP_BYTES = 4 * 1024 * 1024 * 1024
_WORKER_TIMEOUT_STOP_SECONDS = 3


def _worker_memory_max_bytes() -> int:
    """Return a finite per-worker cgroup limit without widening host risk.

    ``TERMINAL_LOCAL_MEMORY_MAX_MB`` may tighten the calculated bound, but an
    oversized value cannot widen it. Otherwise use the tightest known bound
    from the current cgroup and half of physical RAM, capped at 4 GiB.
    """
    override_bound: Optional[int] = None
    override = os.getenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "").strip()
    if override:
        override_valid = False
        try:
            parsed = int(override) * 1024 * 1024
            if parsed >= _MIN_WORKER_MEMORY_MAX_BYTES:
                override_bound = parsed
                override_valid = True
        except ValueError:
            pass
        if not override_valid:
            logger.warning(
                "Ignoring invalid TERMINAL_LOCAL_MEMORY_MAX_MB=%r; "
                "expected an integer representing at least %d MiB",
                override,
                _MIN_WORKER_MEMORY_MAX_BYTES // (1024 * 1024),
            )

    candidates: List[int] = []
    try:
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
            if line.startswith("0::"):
                relative = line.partition("::")[2].lstrip("/")
                raw_limit = (Path("/sys/fs/cgroup") / relative / "memory.max").read_text(
                    encoding="utf-8"
                ).strip()
                if raw_limit.isdigit():
                    cgroup_limit = int(raw_limit)
                    if cgroup_limit >= _MIN_WORKER_MEMORY_MAX_BYTES:
                        candidates.append(cgroup_limit)
                break
    except (OSError, ValueError):
        pass

    try:
        physical_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(
            os.sysconf("SC_PAGE_SIZE")
        )
        physical_bound = min(
            _WORKER_MEMORY_MAX_CAP_BYTES,
            max(_MIN_WORKER_MEMORY_MAX_BYTES, physical_bytes // 2),
        )
        candidates.append(physical_bound)
    except (OSError, ValueError, TypeError):
        pass

    safe_bound = min(candidates) if candidates else _DEFAULT_WORKER_MEMORY_MAX_BYTES
    return min(override_bound, safe_bound) if override_bound else safe_bound


def _systemd_run_user_scope_available() -> bool:
    """Return True when ``systemd-run --user --scope`` is usable.

    ``shutil.which`` is intentionally not sufficient: some environments ship the
    binary on PATH without a usable user bus. This function performs a short,
    cached probe command so callers do not repeatedly hit the failure path.
    """
    global _SYSTEMD_SCOPE_AVAILABLE, _SYSTEMD_SCOPE_PROBED_AT
    cached = _SYSTEMD_SCOPE_AVAILABLE
    now = time.monotonic()
    if cached is True:
        return True
    if (
        cached is False
        and now - _SYSTEMD_SCOPE_PROBED_AT < _SYSTEMD_SCOPE_FAILURE_TTL_SECONDS
    ):
        return False

    with _SYSTEMD_SCOPE_PROBE_LOCK:
        cached = _SYSTEMD_SCOPE_AVAILABLE
        now = time.monotonic()
        if cached is True:
            return True
        if (
            cached is False
            and now - _SYSTEMD_SCOPE_PROBED_AT < _SYSTEMD_SCOPE_FAILURE_TTL_SECONDS
        ):
            return False

        available = False
        if not _IS_WINDOWS:
            try:
                binary = shutil.which("systemd-run")
                if binary:
                    probe_unit = (
                        f"hermes-probe-scope-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                    )
                    result = subprocess.run(
                        [
                            binary,
                            "--user",
                            "--scope",
                            "--quiet",
                            "--unit",
                            probe_unit,
                            "--collect",
                            "--property",
                            "MemoryAccounting=yes",
                            "--property",
                            f"MemoryMax={_worker_memory_max_bytes()}",
                            "--property",
                            "OOMPolicy=kill",
                            "--",
                            "/bin/true",
                        ],
                        capture_output=True,
                        timeout=3,
                    )
                    available = result.returncode == 0
                    if not available:
                        logger.debug(
                            "systemd-run --user --scope probe failed (rc=%s): %s",
                            result.returncode,
                            (result.stderr or b"").decode(
                                "utf-8", "replace"
                            ).strip(),
                        )
            except Exception as exc:
                logger.debug("systemd-run --user --scope probe error: %s", exc)
                available = False

        _SYSTEMD_SCOPE_AVAILABLE = available
        _SYSTEMD_SCOPE_PROBED_AT = time.monotonic()
        return available


def _is_supervised_gateway_process() -> bool:
    """Return whether this process is the live supervised gateway process."""
    if os.environ.get("_HERMES_GATEWAY") != "1":
        return False

    try:
        from gateway.restart import is_gateway_supervisor_process
        from gateway.status import get_running_pid

        return (
            is_gateway_supervisor_process()
            and get_running_pid(cleanup_stale=False) == os.getpid()
        )
    except Exception as exc:
        logger.debug("Could not verify supervised gateway process identity: %s", exc)
        return False


def _prepare_systemd_scope_argv(
    shell_argv: List[str], unit_suffix: str
) -> tuple[List[str], str]:
    """Return ``(argv, unit_name)`` for a scoped launch.

    ``unit_name`` is empty when the systemd-run binary disappeared after a
    successful capability probe. Callers must only record scope metadata when
    this function returns a non-empty unit name.
    """
    binary = shutil.which("systemd-run")
    if binary is None:
        return shell_argv, ""

    unit_name = f"hermes-worker-{unit_suffix}"
    memory_max = _worker_memory_max_bytes()
    argv = [
        binary,
        "--user",
        "--scope",
        "--quiet",
        "--unit",
        unit_name,
        "--collect",
        "--property",
        "MemoryAccounting=yes",
        "--property",
        f"MemoryMax={memory_max}",
        "--property",
        "OOMPolicy=kill",
        "--property",
        f"TimeoutStopSec={_WORKER_TIMEOUT_STOP_SECONDS}s",
        "--",
        *shell_argv,
    ]
    return argv, f"{unit_name}.scope"


def _build_systemd_scope_argv(shell_argv: List[str], unit_suffix: str) -> List[str]:
    """Compatibility wrapper returning only the scoped command argv."""
    argv, _unit_name = _prepare_systemd_scope_argv(shell_argv, unit_suffix)
    return argv


def _stop_systemd_unit(unit_name: str) -> bool:
    """Stop a transient scope and do not return while descendants can run.

    Worker scopes carry a short ``TimeoutStopSec``. If the synchronous stop
    still exceeds our outer bound, force-kill every process in the cgroup and
    verify that systemd no longer reports the unit active before returning.
    """
    binary = shutil.which("systemctl")
    if binary is None:
        return False

    try:
        result = subprocess.run(
            [binary, "--user", "stop", unit_name],
            capture_output=True,
            timeout=_WORKER_TIMEOUT_STOP_SECONDS + 5,
        )
        if result.returncode != 0:
            stderr = (result.stderr or b"").decode(errors="replace").strip()
            stderr_lower = stderr.lower()
            if any(
                marker in stderr_lower
                for marker in ("not loaded", "not found", "does not exist")
            ):
                return True
            logger.debug(
                "systemctl --user stop %s exited %d: %s",
                unit_name,
                result.returncode,
                stderr,
            )
            return False
        return True
    except subprocess.TimeoutExpired:
        logger.warning(
            "Timed out stopping %s gracefully; forcing SIGKILL for the scope",
            unit_name,
        )
        try:
            kill_result = subprocess.run(
                [
                    binary,
                    "--user",
                    "kill",
                    "--kill-whom=all",
                    "--signal=SIGKILL",
                    unit_name,
                ],
                capture_output=True,
                timeout=3,
            )
            if kill_result.returncode != 0:
                logger.debug(
                    "systemctl --user kill %s exited %d: %s",
                    unit_name,
                    kill_result.returncode,
                    (kill_result.stderr or b"").decode(
                        "utf-8", "replace"
                    ).strip(),
                )
                return False
            state = subprocess.run(
                [binary, "--user", "is-active", unit_name],
                capture_output=True,
                timeout=3,
            )
            state_name = (state.stdout or b"").decode(
                "utf-8", "replace"
            ).strip().lower()
            if state.returncode in {3, 4} and state_name in {
                "inactive",
                "failed",
                "unknown",
            }:
                return True
            logger.debug(
                "Could not verify %s inactive after SIGKILL (rc=%d, state=%r): %s",
                unit_name,
                state.returncode,
                state_name,
                (state.stderr or b"").decode("utf-8", "replace").strip(),
            )
            return False
        except Exception as exc:
            logger.debug("Forced cleanup for %s failed: %s", unit_name, exc)
            return False
    except Exception as exc:
        logger.debug("systemctl --user stop %s failed: %s", unit_name, exc)
        return False
