"""Private executable worker. No pickle, live parent agents or DB handles."""
import socket
import sys

from agent.isolated_protocol import AgentConstruction, check_peer, receive, send


def construct(request, credential=None, *, lease=None, commissioning_context=None, metadata=None, task_cwd=None, task_id=None, **callbacks):
    if task_cwd is not None:
        from pathlib import Path
        from agent.runtime_cwd import set_session_cwd
        from tools.terminal_tool import record_session_cwd
        if not isinstance(task_cwd, str) or not Path(task_cwd).is_absolute() or not Path(task_cwd).is_dir():
            raise ValueError('invalid isolated task directory')
        set_session_cwd(task_cwd)
        record_session_cwd(task_id, task_cwd)
    from run_agent import AIAgent
    from hermes_state import SessionDB
    from agent.model_metadata import is_local_endpoint
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from agent.isolated_protocol import validate_metadata
    from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context


    metadata = validate_metadata(metadata or {})
    if lease is not None:
        from agent.workload_context import install_worker_context
        install_worker_context(lease, commissioning_context)

    kwargs = dict(request.kwargs)
    kwargs.pop('session_cwd', None)  # Applied as runtime context, not an AIAgent argument.
    if credential:
        kwargs['api_key'] = credential
    elif is_local_endpoint(kwargs.get('base_url') or ''):
        kwargs['api_key'] = 'local-no-credential'
    else:
        runtime = resolve_runtime_provider(requested=kwargs.get('provider'),
            explicit_base_url=kwargs.get('base_url'), target_model=kwargs.get('model'))
        kwargs['api_key'] = runtime.get('api_key')
    db = SessionDB()
    try:
        scope = delegated_child_context if request.kind == 'delegation' else non_dispatcher_owned_context
        with scope():
            agent = AIAgent(**kwargs, session_db=db, **callbacks)
        agent._owns_session_db = True
        for key, value in metadata.items():
            setattr(agent, key, value)
        from tools.delegate_tool_config import _resolve_child_credential_pool
        agent._credential_pool = _resolve_child_credential_pool(
            kwargs.get('provider'), None, kwargs.get('base_url'))
        if request.kind == 'delegation':
            from tools.delegate_tool import _apply_child_cache_ttl, _apply_child_compression_cap
            from hermes_cli.config_effective import load_user_config_effective
            _apply_child_cache_ttl(agent)
            _apply_child_compression_cap(agent, load_user_config_effective().get('delegation', {}))
            if kwargs.get('parent_session_id'):
                agent._session_init_model_config['_delegate_from'] = kwargs['parent_session_id']
        return agent
    except BaseException:
        db.close()
        raise


def main(endpoint):
    token = sys.stdin.readline(66).strip()
    if len(token) != 64:
        raise ValueError('missing private bootstrap token')
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(30)
    sock.connect(endpoint)
    check_peer(sock)
    send(sock, {'v': 1, 'op': 'hello', 'token': token})
    from agent.isolated_channel import Channel
    import threading
    import os

    agent = None
    running = threading.Lock()
    finished = threading.Event()
    steering_lock = threading.Lock()

    def handle(op, payload):
        nonlocal agent
        if op == 'steer':
            with steering_lock:
                return bool(agent is not None and not finished.is_set() and agent.steer(payload))
        if op == 'activity':
            return agent.get_activity_summary() if agent is not None else {}
        if op != 'run' or not running.acquire(blocking=False):
            raise ValueError('invalid or repeated agent run')
        try:
            from agent.isolated_routing import install_profile
            install_profile(payload['profile_context'])
            from tools.terminal_tool import set_approval_callback
            from agent.isolated_approval import install
            raw = payload['request']
            request = AgentConstruction(raw['kind'], raw['kwargs'])
            install('cron' if request.kind == 'cron-agent' else payload['approval_policy'])
            from agent.isolated_routing import install as install_routing
            install_routing(payload['routing_context'])
            set_approval_callback(lambda command, description, **options: channel.call(
                'approval', {'command': command, 'description': description, 'options': options}, timeout=300))
            agent = construct(request, credential=payload.get('credential'),
                commissioning_context=payload.get('commissioning_context'),
                lease=payload.get('lease'), metadata=payload.get('metadata'),
                task_cwd=payload['task_cwd'], task_id=payload.get('task_id'),
                tool_progress_callback=lambda *args, **kwargs: channel.call('progress', {'args': args, 'kwargs': kwargs}),
                thinking_callback=lambda text: channel.call('thinking', text))
            from agent.delegation_context import delegated_child_context, non_dispatcher_owned_context
            scope = delegated_child_context if request.kind == 'delegation' else non_dispatcher_owned_context
            with scope():
                from agent.isolated_protocol import validate_run_options
                options = validate_run_options(payload.get('run_options', {}))
                stream = (lambda *args, **kwargs: channel.call('stream',
                    {'args': args, 'kwargs': kwargs})) if payload.get('stream') else None
                result = agent.run_conversation(user_message=payload['user_message'],
                    task_id=payload.get('task_id'), stream_callback=stream, **options)
            with steering_lock:
                finished.set()
                late = agent._drain_pending_steer()
                if late:
                    result['pending_steer'] = '\n'.join(filter(None, [result.get('pending_steer'), late]))
            channel.call('activity', agent.get_activity_summary())
            from agent.isolated_protocol import RESULT_FIELDS
            channel.call('metrics', {key: getattr(agent, key, None) for key in RESULT_FIELDS})
            return result
        finally:
            finished.set()

    # An orphan cannot keep running with its approval/control authority gone.
    def disconnected():
        if not finished.is_set():
            os._exit(70)
    sock.settimeout(None)
    channel = Channel(sock, token, handle, on_disconnect=disconnected)
    channel.closed.wait()
    if agent is not None:
        agent.close()


if __name__ == '__main__':
    main(sys.argv[1])
