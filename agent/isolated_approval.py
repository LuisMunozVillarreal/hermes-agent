"""Human approval relay in the commissioning session, never the worker's UI."""
from tools import approval_context as context


def capture(kind):
    """Only commissioning context may select unattended policy, never a tool payload."""
    if kind == 'cron-agent' or context._is_cron_approval_context():
        return 'cron'
    if context._is_unattended_platform_approval_context():
        return 'unattended'
    if context._is_single_query_approval_context():
        return 'single_query'
    return 'interactive'


def install(policy):
    import os
    from gateway.session_context import set_session_vars
    if policy not in {'cron', 'unattended', 'single_query', 'interactive'}:
        raise ValueError('invalid isolated approval context')
    set_session_vars(platform='api_server' if policy == 'unattended' else '',
                     cron_session='1' if policy == 'cron' else '')
    # This private process has one owner; mask unrelated inherited session flags.
    os.environ['HERMES_SINGLE_QUERY_SESSION'] = '1' if policy == 'single_query' else ''
    os.environ['HERMES_GATEWAY_SESSION'] = ''
    context.set_hermes_interactive_context(True)


def decide(payload, callback=None, *, policy=None):
    from tools import approval
    from tools.approval_gateway_wait import _await_gateway_decision
    from tools.approval_prompt import _present_with_selected_transport
    modes = {'cron': context._get_cron_approval_mode,
             'unattended': context._get_unattended_approval_mode,
             'single_query': context._get_single_query_approval_mode}
    policy = policy or capture('delegation')
    if policy in modes:
        return 'once' if modes[policy]() == 'approve' else 'deny'
    if policy != 'interactive':
        return 'deny'
    options = payload.get('options', {})
    command, description = payload['command'], payload['description']
    session_key = context.get_current_session_key()
    keys = payload.get('pattern_keys', ['isolated-operation'])
    attempt = _present_with_selected_transport(
        command=command, description=description, pattern_key=keys[0], pattern_keys=keys,
        session_key=session_key, surface='isolated-agent',
        allow_session=options.get('allow_session', True), allow_permanent=options.get('allow_permanent', True))
    if attempt.get('selected') and not (attempt.get('failure') and attempt.get('fallback') == 'builtin'):
        return attempt.get('choice') if not attempt.get('failure') else 'deny'
    if context._is_gateway_approval_context():
        notify = approval._gateway_notify_cb(session_key)
        if notify is None:
            return 'deny'
        decision = _await_gateway_decision(session_key, notify, {
            'command': command, 'description': description, 'pattern_key': keys[0],
            'pattern_keys': keys, **options}, surface='isolated-agent')
        return decision.get('choice') if decision.get('resolved') else 'deny'
    if callback is None:
        from tools.delegate_tool_config import _get_subagent_approval_callback
        callback = _get_subagent_approval_callback()
    return callback(command, description, **options)
