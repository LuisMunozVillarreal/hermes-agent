"""Routing comes from commissioning context, not agent/model arguments."""
import contextvars

import pytest

from agent.isolated_routing import capture, install, validate


@pytest.mark.parametrize('fault', ['extra-authority', 'capability-type', 'oversize', 'not-string'])
def test_routing_rejects_untrusted_fields_and_values(fault):
    context = capture()
    if fault == 'extra-authority':
        context['session']['browser_control_principal'] = 'forged'
    elif fault == 'capability-type':
        context['async_delivery'] = 'true'
    elif fault == 'oversize':
        context['delivery']['chat_id'] = 'é' * 4096
    else:
        context['session']['chat_id'] = {'forged': 'route'}
    with pytest.raises(ValueError, match='invalid isolated'):
        validate(context)


def test_routing_preserves_false_capability_without_exporting_wake_authority():
    from gateway.session_context import (set_session_vars, async_delivery_supported,
        session_history_delivery_supported, get_session_env, _VAR_MAP)
    def commissioning():
        set_session_vars(platform='telegram', chat_id='123', thread_id='456',
            async_delivery=False, session_history_delivery='1', browser_control_principal='private')
        _VAR_MAP['HERMES_CRON_AUTO_DELIVER_PLATFORM'].set('telegram')
        _VAR_MAP['HERMES_CRON_AUTO_DELIVER_CHAT_ID'].set('123')
        _VAR_MAP['HERMES_CRON_AUTO_DELIVER_THREAD_ID'].set('456')
        return capture()
    captured = contextvars.Context().run(commissioning)
    def worker():
        from agent.isolated_approval import install as install_approval
        install_approval('cron')
        install(captured)
        assert not async_delivery_supported()
        assert not session_history_delivery_supported()
        assert get_session_env('HERMES_CRON_SESSION') == '1'
        assert get_session_env('HERMES_BROWSER_CONTROL_PRINCIPAL') == ''
        assert get_session_env('HERMES_SESSION_CHAT_ID') == '123'
        assert get_session_env('HERMES_CRON_AUTO_DELIVER_THREAD_ID') == '456'
    contextvars.Context().run(worker)
