"""Real app discovery -> Unix coordinator -> isolated AIAgent -> nested terminal.

Requires the standalone hermes-worker-queue package in the test venv and a
working user-systemd/cgroup-v2 delegation. No remote model or pressure fixture.
"""

import importlib.util
import json
import os
from pathlib import Path
import select
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

MIB = 1024 * 1024
AGENT_CAP = 1024 * MIB
TOOL_CAP = 64 * MIB
BUDGET = AGENT_CAP + 2 * TOOL_CAP


def wait_for(predicate, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    pytest.fail("integration condition timed out")


@pytest.fixture
def queue_home(tmp_path, monkeypatch):
    if not sys.platform.startswith("linux"):
        pytest.skip("requires Linux user-systemd/cgroup v2")
    spec = importlib.util.find_spec("hermes_worker_queue")
    if spec is None:
        pytest.skip("install the paired out-of-tree app in the isolated test venv")
    from hermes_cli.plugins import get_plugin_manager
    from agent.workload_registry import snapshot_registration
    from hermes_worker_queue.client import Client
    from hermes_worker_queue.enforcement.linux import Systemd

    source = Path(spec.origin).parents[2]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.setenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
    monkeypatch.setenv(
        "DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus"
    )
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "hermes-worker-queue").symlink_to(
        source, target_is_directory=True
    )
    with tempfile.TemporaryDirectory(prefix="hai-queue-", dir="/tmp") as directory:
        state = Path(directory)
        (tmp_path / "config.yaml").write_text(
            "model:\n  default: fixture\ncompression:\n  enabled: false\n"
            "plugins:\n  enabled: [hermes-worker-queue]\n"
            "workloads:\n  mode: required\n  provider: hermes-worker-queue\n"
            f"  state: {state}\n  peak_memory_bytes: {AGENT_CAP}\n"
            f"  estimates:\n    terminal: {TOOL_CAP}\n"
            # This forged configuration must be discarded, not inherited by tools.
            "  parent_lease:\n    ticket: 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\n"
            "    generation: 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'\n"
        )
        # AIAgent also discovers plugins: share its profile-keyed manager rather
        # than registering the same provider through an independent manager.
        manager = get_plugin_manager()
        manager.discover_and_load()
        provider = snapshot_registration("hermes-worker-queue")
        assert provider is not None
        coordinator = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "hermes_worker_queue.cli",
                "--state",
                str(state),
                "serve",
                "--budget",
                str(BUDGET),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        client = Client(state)
        try:
            assert select.select([coordinator.stdout], [], [], 10)[0]
            assert coordinator.stdout.readline().strip() == "ready"
            yield tmp_path, state, client, provider, Systemd()
        finally:
            coordinator.terminate()
            _, errors = coordinator.communicate(timeout=15)
            manager.unload()
            assert coordinator.returncode == 0, errors
            assert snapshot_registration("hermes-worker-queue") is None


@pytest.fixture
def gated_model():
    """Pause the first local completion until the test has checked admission."""
    entered, release = threading.Event(), threading.Event()
    requests = []
    settings = {"command": ""}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if "messages" not in body:
                self.send_error(400, "fixture supports chat completions only")
                return
            requests.append(body)
            message = {"role": "assistant", "content": "queue-contained-result"}
            finish = "stop"
            if not any(m["role"] == "tool" for m in body["messages"]):
                entered.set()
                if not release.wait(30):
                    self.send_error(504)
                    return
                finish = "tool_calls"
                message = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "nested-terminal",
                            "type": "function",
                            "function": {
                                "name": "terminal",
                                "arguments": json.dumps({
                                    "command": settings["command"],
                                    "timeout": 60,
                                }),
                            },
                        }
                    ],
                }
            response = {
                "id": "local-queue-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
            if body.get("stream"):
                response["object"] = "chat.completion.chunk"
                response["choices"][0]["delta"] = response["choices"][0].pop("message")
                payload = (
                    "data: " + json.dumps(response) + "\n\ndata: [DONE]\n\n"
                ).encode()
            else:
                payload = json.dumps(response).encode()
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/event-stream" if body.get("stream") else "application/json",
            )
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            f"http://127.0.0.1:{server.server_port}/v1",
            settings,
            entered,
            release,
            requests,
        )
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(3)


@pytest.mark.parametrize("kind", ["delegation", "cron-agent"])
@pytest.mark.parametrize("outcome", ["complete", "cancel"])
def test_real_agent_nested_terminal_growth_and_cleanup(
    queue_home, gated_model, kind, outcome
):
    from agent.isolated_agent import IsolatedAgent

    home, state, client, provider, backend = queue_home
    url, settings, entered, release, requests = gated_model
    marker, finish = home / "terminal-cgroup", home / "finish-terminal"
    settings["command"] = (
        f"cat /proc/self/cgroup > {shlex.quote(str(marker))}; "
        f"while [ ! -f {shlex.quote(str(finish))} ]; do sleep 0.05; done; "
        "printf nested-queue-output"
    )
    child = IsolatedAgent(
        kind=kind,
        model="fixture",
        provider="custom",
        base_url=url,
        enabled_toolsets=["terminal"],
        skip_memory=True,
        skip_context_files=True,
        max_iterations=3,
        quiet_mode=True,
    )
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(
            child.run_conversation, "execute the local fixture command"
        )
        try:
            assert entered.wait(25), (
                future.result(timeout=1)
                if future.done()
                else "worker never called local model"
            )
            root = client.call("status")["jobs"][0]
            assert root["kind"] == kind and root["state"] == "running"
            assert root["parent_ticket"] is None  # forged config was stripped
            assert child.process.workload_lease == {
                "ticket": root["ticket"],
                "generation": root["generation"],
            }
            assert root["unit"] in Path(f"/proc/{child.worker_pid}/cgroup").read_text()
            assert child.worker_pid != os.getpid()
            cap = backend.path(root["unit"]) / "memory.max"
            assert int(cap.read_text()) == AGENT_CAP
            assert client.call("status")["reserved"] == AGENT_CAP

            # Absolute growth adds child slack, not another independent reservation.
            config = {"state": str(state)}
            retained_target = AGENT_CAP + 16 * MIB
            target = retained_target + TOOL_CAP
            for _ in range(2):
                result = provider.request_growth(
                    child.process.workload_lease,
                    target,
                    retained_memory_bytes=retained_target,
                    config=config,
                )
                assert result["granted"] is True
                assert result["job"]["peak"] == target
            denied = provider.request_growth(
                child.process.workload_lease,
                BUDGET + MIB,
                retained_memory_bytes=retained_target,
                config=config,
            )
            assert denied["granted"] is False
            assert int(cap.read_text()) == retained_target
            release.set()
            wait_for(marker.exists)
            status = client.call("status")
            descendants = [r for r in status["jobs"] if r["ticket"] != root["ticket"]]
            # LocalEnvironment also launches a short shell-initialization probe.
            assert descendants and all(
                r["parent_ticket"] == root["ticket"]
                and r["parent_generation"] == root["generation"]
                for r in descendants
            ), status
            nested = next(r for r in descendants if r["unit"] in marker.read_text())
            assert nested["parent_generation"] == root["generation"]
            assert nested["kind"] == "terminal" and nested["state"] == "running"
            assert nested["unit"] in marker.read_text()
            assert (
                int((backend.path(nested["unit"]) / "memory.max").read_text())
                == TOOL_CAP
            )
            assert status["reserved"] == target
            admitted_root = next(
                r for r in status["jobs"] if r["ticket"] == root["ticket"]
            )
            assert admitted_root["retained"] + nested["peak"] == target
            if outcome == "cancel":
                assert child.interrupt("integration cancellation")
                result = future.result(timeout=10)
                assert result["interrupted"] is True
                assert not result.get("completed")
            else:
                finish.touch()
                result = future.result(timeout=15)
                assert result["final_response"] == "queue-contained-result", result
                tools = [
                    m for r in requests for m in r["messages"] if m["role"] == "tool"
                ]
                assert any("nested-queue-output" in m["content"] for m in tools), tools
            wait_for(lambda: client.call("status")["reserved"] == 0)
            assert backend.empty(root["unit"])
            assert all(backend.empty(r["unit"]) for r in descendants)
            assert child.process.poll() is not None
        finally:
            release.set()
            finish.touch()
            if not future.done():
                child.interrupt("fixture cleanup")
            child.close()


@pytest.mark.parametrize("kind", ["delegation", "cron-agent"])
def test_impossible_nested_budget_fails_closed_without_shell(
    queue_home, gated_model, kind
):
    from agent.isolated_agent import IsolatedAgent

    home, _state, client, _provider, backend = queue_home
    url, settings, entered, release, requests = gated_model
    marker = home / "must-not-execute"
    settings["command"] = f"printf bypass > {shlex.quote(str(marker))}"
    # The agent fits; its retained allocation plus the requested terminal does not.
    config = home / "config.yaml"
    config.write_text(
        config.read_text().replace(
            f"terminal: {TOOL_CAP}", f"terminal: {2 * TOOL_CAP + MIB}"
        )
    )
    child = IsolatedAgent(
        kind=kind,
        model="fixture",
        provider="custom",
        base_url=url,
        enabled_toolsets=["terminal"],
        skip_memory=True,
        skip_context_files=True,
        max_iterations=3,
        quiet_mode=True,
    )
    release.set()
    try:
        result = child.run_conversation("execute the fixture command")
        assert entered.is_set()
        tools = [m for r in requests for m in r["messages"] if m["role"] == "tool"]
        assert tools, result
        assert any("error" in m["content"].lower() for m in tools), tools
        assert not marker.exists(), "nested terminal bypassed denied admission"
        wait_for(lambda: client.call("status")["reserved"] == 0)
        jobs = client.call("status")["jobs"]
        assert len(jobs) == 1 and jobs[0]["kind"] == kind, jobs
        assert backend.empty(jobs[0]["unit"])
    finally:
        child.close()
