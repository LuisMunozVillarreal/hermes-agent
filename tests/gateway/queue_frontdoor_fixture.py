"""Disposable, credential-free frontdoor clients for the queue acceptance test."""
import asyncio
import json
import os
from pathlib import Path
import sys


def gateway(url):
    from gateway.run import GatewayRunner
    from gateway.config import load_gateway_config, Platform
    from gateway.session import SessionSource
    from gateway.platforms.event import MessageEvent
    from gateway.platforms.base import BasePlatformAdapter

    class LocalTransport:
        extract_media = staticmethod(BasePlatformAdapter.extract_media)
        extract_images = staticmethod(BasePlatformAdapter.extract_images)

        def toolsets_for_source(self, source):
            return ["terminal"]

        async def send(self, chat_id, content, **kwargs):
            Path(os.environ["HERMES_HOME"], "gateway-reply").write_text(content)
            return {"success": True, "message_id": "fixture"}

    async def run():
        # Initialize actual runner state, except startup's optional downloader.
        # No adapter is connected and no live gateway lifecycle is touched.
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = load_gateway_config()
        runner.adapters = {Platform.TELEGRAM: LocalTransport()}
        runner._profile_adapters = {}
        runner._session_db_init_error = None
        for method in ("_init_runtime_settings", "_init_session_store", "_init_lifecycle_state",
                       "_init_runtime_caches", "_init_session_db", "_init_registries_and_clocks"):
            getattr(runner, method)()
        try:
            event = MessageEvent(text="/bg gateway fixture terminal", source=SessionSource(
                platform=Platform.TELEGRAM, user_id="fixture-user", chat_id="fixture-chat"))
            await runner._handle_background_command(event)
            await asyncio.gather(*tuple(runner._background_tasks))
            response = Path(os.environ["HERMES_HOME"], "gateway-reply").read_text()
            assert "queue-contained-result" in response, response
        finally:
            if runner._executor:
                runner._executor.shutdown(wait=True)
            if runner._session_db:
                runner._session_db.close()
    asyncio.run(run())


def cli(url):
    from cli import main
    main(query="CLI fixture terminal", oneshot=True, toolsets="terminal", model="fixture",
         provider="custom", api_key="local-fixture", base_url=url, quiet=True, max_turns=3)


def cron(url):
    from cron.jobs import create_job
    from cron.scheduler_provider import InProcessCronScheduler
    home = Path(os.environ["HERMES_HOME"])
    scripts = home / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / "frontdoor.py").write_text(
        "from pathlib import Path\nimport time\n"
        f"home=Path({str(home)!r})\n(home/'cron-ran').touch()\n"
        "while not (home/'release').exists(): time.sleep(.05)\nprint('cron-result')\n")
    job = create_job(prompt="", schedule="every 1h", script="frontdoor.py",
                     no_agent=True, deliver="local")
    assert InProcessCronScheduler().fire_due(job["id"])


if __name__ == "__main__":
    from hermes_cli.plugins import get_plugin_manager
    get_plugin_manager().discover_and_load()
    {"gateway": gateway, "cli": cli, "cron": cron}[sys.argv[1]](sys.argv[2])
