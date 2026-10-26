"""Managed PTY creation for the generic process registry."""
import os


def spawn_managed_pty(self, command, task_id, owner_task_id, session_key, cwd, env_vars, persist_on_release):
    from agent.workload_boundary import execution_provider, popen
    from tools.process_registry import _resolve_safe_cwd, _find_shell
    execution_provider("terminal-pty")
    session = self._new_session(command, task_id, owner_task_id, session_key, _resolve_safe_cwd(cwd or os.getcwd()),
                                persist_on_release=persist_on_release)
    session._pty = popen("terminal-pty", [_find_shell(), "-lic", f"set +m; {command}"],
                         cwd=session.cwd, env=self._spawn_env(env_vars))
    session.pid = session._pty.pid
    session.host_start_time = self._safe_host_start_time(session.pid)
    try:
        self._track_started(session, self._pty_reader_loop, f"proc-pty-reader-{session.id}")
    except BaseException:
        session._pty.terminate(force=True)
        session._pty.close()
        raise
    return session
