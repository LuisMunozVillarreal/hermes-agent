"""Allowlisted commissioning context carried only by authenticated bootstrap.

Routing is not construction configuration. In particular browser principals,
parent session keys and server-history wake authority never cross this boundary.
"""
SESSION_FIELDS = ('platform', 'source', 'chat_id', 'chat_type', 'chat_name',
    'thread_id', 'user_id', 'user_id_alt', 'user_name', 'scope_id', 'message_id',
    'profile', 'parent_chat_id')
DELIVERY_FIELDS = ('platform', 'chat_id', 'thread_id')

# Only process plumbing crosses exec. Credentials and behavioral context travel
# over authenticated IPC, never the launcher's argv/environment/queue ledger.
_WORKER_ENV = frozenset('''PATH HOME USER LOGNAME LANG LC_ALL LC_CTYPE TZ SHELL
TMPDIR VIRTUAL_ENV PYTHONPATH PYTHONUTF8 PYTHONDONTWRITEBYTECODE PYTHONHASHSEED
SSL_CERT_FILE SSL_CERT_DIR XDG_RUNTIME_DIR DBUS_SESSION_BUS_ADDRESS
HERMES_HOME HERMES_REAL_HOME'''.split())


def capture_profile():
    import os
    from pathlib import Path
    from hermes_constants import get_hermes_home, get_process_hermes_home
    from agent.secret_scope import (current_secret_scope, is_multiplex_active,
        build_profile_secret_scope, _is_global_env, UnscopedSecretError)
    from tools.terminal_scope import (get_terminal_scope, enforce_no_refusal,
        build_profile_terminal_scope)
    from tools.environments.local import build_subprocess_env
    from agent.delegation_context import scrub_kanban_env

    home = get_hermes_home().expanduser().resolve(strict=True)
    scope = current_secret_scope()
    multiplex = is_multiplex_active()
    if multiplex and scope is None:
        raise UnscopedSecretError('isolated commissioning requires a profile secret scope')
    foreign_home = home != get_process_hermes_home().expanduser().resolve()
    restricted = multiplex or foreign_home
    if foreign_home and scope is None:
        scope = build_profile_secret_scope(home)
    scope = None if scope is None else dict(scope)
    enforce_no_refusal()
    terminal = get_terminal_scope()
    if terminal is None and restricted:
        terminal = build_profile_terminal_scope(home)
    terminal = None if terminal is None else dict(terminal)
    env = {k: v for k, v in os.environ.items() if not restricted or k in _WORKER_ENV}
    if scope is not None:
        env.update((k, v) for k, v in scope.items() if not _is_global_env(k))
    # A profile label is display/routing metadata, not home-selection authority.
    env.pop('HERMES_PROFILE', None)
    env['HERMES_HOME'] = str(home)
    if terminal is not None:
        env = {k: v for k, v in env.items() if not k.startswith('TERMINAL_')}
        env.update(terminal)
        env.setdefault('TERMINAL_HOME_MODE', 'auto')
    env = scrub_kanban_env(build_subprocess_env(env, scrub_secrets=False))
    stat = home.stat()
    return validate_profile({'home': str(home), 'identity': [stat.st_dev, stat.st_ino],
        'multiplex': multiplex, 'secrets': scope, 'environment': env, 'terminal': terminal})


def validate_profile(profile):
    from pathlib import Path
    if not isinstance(profile, dict) or set(profile) != {
            'home', 'identity', 'multiplex', 'secrets', 'environment', 'terminal'}:
        raise ValueError('invalid isolated profile context')
    home = profile['home']
    if not isinstance(home, str) or '\x00' in home or not Path(home).is_absolute():
        raise ValueError('invalid isolated profile home')
    path = Path(home)
    if not path.is_dir() or str(path.resolve()) != home:
        raise ValueError('invalid isolated profile directory')
    identity = profile['identity']
    stat = path.stat()
    if (not isinstance(identity, list) or len(identity) != 2
            or any(type(v) is not int for v in identity)
            or identity != [stat.st_dev, stat.st_ino]):
        raise ValueError('isolated profile directory identity changed')
    if type(profile['multiplex']) is not bool:
        raise ValueError('invalid isolated multiplex mode')
    for key in ('secrets', 'environment', 'terminal'):
        mapping = profile[key]
        if mapping is None and key != 'environment':
            continue
        if not isinstance(mapping, dict) or any(
                not isinstance(k, str) or not k or '=' in k or '\x00' in k
                or not isinstance(v, str) or '\x00' in v for k, v in mapping.items()):
            raise ValueError('invalid isolated profile mapping')
    if profile['multiplex'] and profile['secrets'] is None:
        raise ValueError('missing isolated profile secret scope')
    if profile['environment'].get('HERMES_HOME') != home:
        raise ValueError('isolated profile environment mismatch')
    if profile['terminal'] is not None and any(
            not k.startswith('TERMINAL_') for k in profile['terminal']):
        raise ValueError('invalid isolated terminal policy')
    from agent.isolated_transport import validate_payload
    validate_payload(profile)
    return profile


def profile_environment(profile):
    profile = validate_profile(profile)
    return {k: v for k, v in profile['environment'].items() if k in _WORKER_ENV}


def install_profile(profile):
    """First operation in the authenticated run handler, before runtime imports."""
    import os
    from hermes_constants import set_hermes_home_override
    from agent.secret_scope import set_secret_scope, set_multiplex_active
    profile = validate_profile(profile)
    if os.environ.get('HERMES_HOME') != profile['home']:
        raise ValueError('isolated launch/bootstrap profile mismatch')
    os.environ.clear()
    os.environ.update(profile['environment'])
    set_hermes_home_override(profile['home'])
    set_secret_scope(profile['secrets'])
    set_multiplex_active(profile['multiplex'])
    from tools.terminal_scope import set_terminal_scope
    set_terminal_scope(profile['terminal'])


def validate(context):
    if not isinstance(context, dict) or set(context) != {'session', 'delivery', 'async_delivery'}:
        raise ValueError('invalid isolated routing context')
    if type(context['async_delivery']) is not bool:
        raise ValueError('invalid isolated delivery capability')
    for section, fields in (('session', SESSION_FIELDS), ('delivery', DELIVERY_FIELDS)):
        values = context[section]
        if not isinstance(values, dict) or set(values) != set(fields):
            raise ValueError('invalid isolated routing fields')
        if any(not isinstance(v, str) or len(v.encode('utf-8')) > 4096 or '\x00' in v
               for v in values.values()):
            raise ValueError('invalid isolated routing value')
    return context


def capture():
    from gateway.session_context import get_session_env, async_delivery_supported
    return validate({
        'session': {key: get_session_env('HERMES_SESSION_' + key.upper()) for key in SESSION_FIELDS},
        'delivery': {key: get_session_env('HERMES_CRON_AUTO_DELIVER_' + key.upper()) for key in DELIVERY_FIELDS},
        'async_delivery': async_delivery_supported(),
    })


def install(context):
    from gateway.session_context import _VAR_MAP, _SESSION_ASYNC_DELIVERY
    context = validate(context)
    # Approval installation owns cron/unattended flags; do not reset them here.
    for section, prefix in (('session', 'HERMES_SESSION_'), ('delivery', 'HERMES_CRON_AUTO_DELIVER_')):
        for key, value in context[section].items():
            _VAR_MAP[prefix + key.upper()].set(value)
    _SESSION_ASYNC_DELIVERY.set(context['async_delivery'])
