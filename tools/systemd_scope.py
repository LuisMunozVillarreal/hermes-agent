"""Shared systemd scope policy for foreground and background execution."""
import logging
import os
import platform
import shutil
import stat
import subprocess
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path
from typing import Dict, List, Optional

_IS_LINUX = platform.system() == "Linux"
_IS_WINDOWS = platform.system() == "Windows"
_WORKER_TIMEOUT_STOP_SECONDS = 3
logger = logging.getLogger(__name__)

_SYSTEMD_SCOPE_AVAILABLE: Optional[bool] = None
_SYSTEMD_SCOPE_PROBE_LOCK = threading.Lock()
_SYSTEMD_SCOPE_PROBED_AT = 0.0
_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS = 60.0
_MIN_WORKER_MEMORY_MAX_BYTES = 64 * 1024 * 1024
_DEFAULT_WORKER_MEMORY_MAX_BYTES = 1024 * 1024 * 1024
_WORKER_MEMORY_MAX_CAP_BYTES = 4 * 1024 * 1024 * 1024


def _worker_memory_max_bytes() -> int:
    """Finite per-worker cgroup limit that can never widen host risk.
    ``TERMINAL_LOCAL_MEMORY_MAX_MB`` is honored only when it *tightens* the safe
    bound (min of the gateway's cgroup-v2 ``memory.max`` and half of physical RAM,
    capped at 4 GiB), so an oversized override cannot exceed the enclosing slice.

    The proposed local-memory-guard environment override is honored when it tightens the safe bound, so this
    isolation composes with PR #57121 instead of inventing a second knob.
    """
    override_bound: Optional[int] = None
    override = os.getenv("TERMINAL_LOCAL_MEMORY_MAX_MB", "").strip()
    if override:
        try:
            parsed = int(override) * 1024 * 1024
        except ValueError:
            parsed = -1
        if parsed >= _MIN_WORKER_MEMORY_MAX_BYTES:
            override_bound = parsed
        else:
            logger.warning(
                "Ignoring invalid TERMINAL_LOCAL_MEMORY_MAX_MB=%r; "
                "expected an integer representing at least %d MiB",
                override, _MIN_WORKER_MEMORY_MAX_BYTES // (1024 * 1024))
    candidates: List[int] = []
    with suppress(OSError, ValueError):
        lines = Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines()
        v2 = next((ln for ln in lines if ln.startswith("0::")), None)
        if v2 is not None:
            relative = v2.partition("::")[2].lstrip("/")
            raw_limit = (Path("/sys/fs/cgroup") / relative / "memory.max").read_text(encoding="utf-8").strip()
            if raw_limit.isdigit() and int(raw_limit) >= _MIN_WORKER_MEMORY_MAX_BYTES:
                candidates.append(int(raw_limit))
    with suppress(OSError, ValueError, TypeError):
        physical_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        candidates.append(min(_WORKER_MEMORY_MAX_CAP_BYTES, max(_MIN_WORKER_MEMORY_MAX_BYTES, physical_bytes // 2)))
    safe_bound = min(candidates) if candidates else _DEFAULT_WORKER_MEMORY_MAX_BYTES
    return min(override_bound, safe_bound) if override_bound else safe_bound


def _systemd_scope_argv(binary: str, unit_name: str, *argv: str) -> List[str]:
    """``systemd-run --user --scope`` argv shared by the probe and real spawns.
    ``--collect`` self-cleans the scope after exit; ``--unit`` names it for systemctl.
    No ``OOMPolicy=``: transient scopes reject it on systemd <253 (#102486)."""
    return [
        binary, "--user", "--scope", "--quiet", "--unit", unit_name, "--collect",
        "--property", "MemoryAccounting=yes",
        "--property", f"MemoryMax={_worker_memory_max_bytes()}",
        "--property", f"TimeoutStopSec={_WORKER_TIMEOUT_STOP_SECONDS}s",
        "--", *argv,
    ]


def _default_user_runtime_dir() -> Path:
    """``/run/user/<uid>``; a function so tests can point it at a temp dir with a real socket."""
    return Path(f"/run/user/{os.getuid()}")  # windows-footgun: ok — only reached behind the _IS_LINUX gate in systemd_user_bus_env


def _secure_user_runtime_dir(path: Path) -> bool:
    """Accept only an absolute, owned, non-writable real directory."""
    try:
        metadata = path.lstat()
        return (
            path.is_absolute()
            and stat.S_ISDIR(metadata.st_mode)
            and metadata.st_uid == os.getuid()  # windows-footgun: ok — only reached behind the _IS_LINUX gate in systemd_user_bus_env
            and metadata.st_mode & 0o022 == 0
        )
    except OSError:
        return False


def systemd_user_bus_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build an environment that can reach this user's lingering systemd manager.

    System-level gateway units run as an unprivileged ``User=`` but normally do
    not inherit login-session variables.  When the conventional runtime
    directory is owned by this uid and its bus exists, derive the two standard
    variables.  Derived fresh on every call rather than adopted once at boot:
    linger may be enabled after the gateway started (existing installs), so
    the bus can appear later and the probe's failure TTL must be able to
    recover (#104893).
    The returned copy is passed explicitly to the probe and every scoped spawn;
    ``os.environ`` is left unchanged.
    """
    env = dict(os.environ if base_env is None else base_env)
    if not _IS_LINUX:
        return env
    configured = env.get("XDG_RUNTIME_DIR")
    if configured and _secure_user_runtime_dir(Path(configured)):
        runtime_dir = Path(configured)
    else:
        runtime_dir = _default_user_runtime_dir()
        if not _secure_user_runtime_dir(runtime_dir):
            return env

    bus_path = runtime_dir / "bus"
    try:
        bus_metadata = bus_path.lstat()
    except OSError:
        return env
    if not stat.S_ISSOCK(bus_metadata.st_mode) or bus_metadata.st_uid != os.getuid():  # windows-footgun: ok — behind the _IS_LINUX gate above
        return env

    env["XDG_RUNTIME_DIR"] = str(runtime_dir)
    env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus_path}"
    return env


def _systemd_scope_cached() -> Optional[bool]:
    """Cached probe verdict, or None when a (re)probe is due. True is permanent; False
    expires after ``_SYSTEMD_SCOPE_FAILURE_TTL_SECONDS`` so a D-Bus blip isn't sticky."""
    if _SYSTEMD_SCOPE_AVAILABLE is True:
        return True
    stale = time.monotonic() - _SYSTEMD_SCOPE_PROBED_AT >= _SYSTEMD_SCOPE_FAILURE_TTL_SECONDS
    return None if _SYSTEMD_SCOPE_AVAILABLE is None or stale else False


def _systemd_run_user_scope_available() -> bool:
    """True if ``systemd-run --user --scope`` can create a cgroup.
    ``shutil.which`` alone is insufficient: system services and containers may lack
    the user D-Bus bus even with the binary on PATH (every spawn would fail with
    ``Failed to connect to user bus``), so a cheap probe is run and cached.

    Use ``/bin/sh -c 'exit 0'``: NixOS provides ``/bin/sh`` but not ``/bin/true``
    (#105365), regardless of the gateway service's PATH."""
    global _SYSTEMD_SCOPE_AVAILABLE, _SYSTEMD_SCOPE_PROBED_AT
    verdict = _systemd_scope_cached()
    if verdict is not None:
        return verdict
    # Double-checked locking: a concurrent first-use spawn must not observe a temporary
    # False mid-probe, or it would launch back inside the gateway cgroup.
    with _SYSTEMD_SCOPE_PROBE_LOCK:
        verdict = _systemd_scope_cached()
        if verdict is not None:
            return verdict
        available = False
        if _IS_LINUX:
            try:
                import shutil

                binary = shutil.which("systemd-run")
                if binary:
                    # Unique unit avoids collisions; the timeout bounds D-Bus.
                    probe_unit = f"hermes-probe-scope-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                    result = subprocess.run(
                        _systemd_scope_argv(binary, probe_unit, "/bin/sh", "-c", "exit 0"),
                        capture_output=True,
                        timeout=3,
                        env=systemd_user_bus_env(),
                    )
                    available = result.returncode == 0
                    if not available:
                        logger.debug(
                            "systemd-run --user --scope probe failed (rc=%s): %s",
                            result.returncode, (result.stderr or b"").decode("utf-8", "replace").strip(),
                        )
            except Exception as exc:
                logger.debug("systemd-run --user --scope probe error: %s", exc)
        _SYSTEMD_SCOPE_AVAILABLE = available
        _SYSTEMD_SCOPE_PROBED_AT = time.monotonic()
        return available


def _is_supervised_gateway_process() -> bool:
    """Whether this process is the live, supervised Hermes gateway itself.
    Supervisor markers and ``_HERMES_GATEWAY`` are inherited by every descendant (and
    importing ``gateway.run`` sets the latter), so also require ownership of the live
    gateway PID file — scopes are for the gateway, not terminal children or CLIs."""
    if os.environ.get("_HERMES_GATEWAY") != "1":
        return False
    try:
        from gateway.restart import is_gateway_supervisor_process
        from gateway.status import get_running_pid

        return is_gateway_supervisor_process() and get_running_pid(cleanup_stale=False) == os.getpid()
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
    argv = _systemd_scope_argv(binary, unit_name, *shell_argv)
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
            stdin=subprocess.DEVNULL, env=systemd_user_bus_env(),
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
                stdin=subprocess.DEVNULL, env=systemd_user_bus_env(),
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
                stdin=subprocess.DEVNULL, env=systemd_user_bus_env(),
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
