"""Correlated duplex RPC; readers never block behind an approval callback."""
import threading
import uuid

import time

from agent.isolated_transport import receive_message, send_message, TRANSFER_TIMEOUT
from agent.isolated_diagnostics import ChannelDisconnected, failure_reason


class Channel:
    def __init__(self, sock, generation, handler, on_disconnect=None):
        self.sock, self.generation, self.handler = sock, generation, handler
        self.on_disconnect = on_disconnect
        self.closed = threading.Event()
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._pending = {}
        self.disconnect_reason = None
        self._slots = threading.BoundedSemaphore(8)
        self._thread = threading.Thread(target=self._read, daemon=True, name='agent-control')
        self._thread.start()

    def _send(self, message):
        deadline = time.monotonic() + TRANSFER_TIMEOUT
        if not self._send_lock.acquire(timeout=TRANSFER_TIMEOUT):
            self.close(reason='agent control connection lost: send queue timed out (TimeoutError)')
            raise ChannelDisconnected(self.disconnect_reason or 'agent control connection lost: local close')
        try:
            if self.closed.is_set():
                raise ChannelDisconnected(self.disconnect_reason or 'agent control connection lost: local close')
            send_message(self.sock, dict(message, v=1, generation=self.generation), deadline)
        except (OSError, EOFError) as exc:
            self.close(reason=f'agent control connection lost: send failed ({failure_reason(exc)})')
            raise ChannelDisconnected(self.disconnect_reason or 'agent control connection lost: local close') from None
        finally:
            self._send_lock.release()

    def call(self, op, payload=None, timeout=30):
        key = uuid.uuid4().hex
        event, box = threading.Event(), {}
        with self._lock:
            if self.closed.is_set():
                raise ChannelDisconnected(self.disconnect_reason or 'agent control connection lost: local close')
            if len(self._pending) >= 8:
                raise RuntimeError('agent control request limit reached')
            self._pending[key] = (event, box)
        try:
            self._send({'id': key, 'op': op, 'payload': payload})
            if not event.wait(timeout):
                raise TimeoutError('agent control acknowledgement timed out')
            if 'error' in box:
                if isinstance(box['error'], ChannelDisconnected):
                    raise ChannelDisconnected(str(box['error']))
                raise RuntimeError(box['error'])
            return box['value']
        finally:
            with self._lock:
                self._pending.pop(key, None)

    def _dispatch(self, message):
        try:
            try:
                value = self.handler(message['op'], message.get('payload'))
                response = {'reply': message['id'], 'value': value}
            except Exception as exc:  # health: allow BLE001 -- RPC boundary serializes arbitrary handler failures to waiting caller
                response = {'reply': message['id'], 'error': f'{type(exc).__name__}: {exc}'}
            try:
                self._send(response)
            except (ValueError, TypeError, RecursionError):
                # Spooling finishes before any bytes are sent. Encoding/total
                # limit failures still settle run(timeout=None) explicitly.
                self._send({'reply': message['id'],
                            'error': 'agent response exceeds IPC limit or is not JSON serializable'})
        except (OSError, EOFError, ValueError, TypeError, RecursionError, ChannelDisconnected) as exc:
            # Release all waiters if even the terminal reply fails.
            self.close(reason=f'agent control connection lost: reply failed ({failure_reason(exc)})')
        finally:
            self._slots.release()

    def _read(self):
        try:
            while not self.closed.is_set():
                message = receive_message(self.sock, self.generation)
                if 'reply' in message:
                    with self._lock:
                        pending = self._pending.get(message['reply'])
                        if pending is not None and not pending[0].is_set():
                            event, box = pending
                            box.update(message)
                            event.set()
                else:
                    if not self._slots.acquire(blocking=False):
                        raise ValueError('agent control concurrency exceeded')
                    threading.Thread(target=self._dispatch, args=(message,), daemon=True).start()
        except (OSError, EOFError, ValueError, KeyError, TypeError, RecursionError) as exc:
            self.close(reason=f'agent control connection lost: receive failed ({failure_reason(exc)})')
        finally:
            self.close()
            if self.on_disconnect:
                self.on_disconnect()

    def close(self, *, reason=None):
        # Publish the first cause and settle waiters atomically. Socket shutdown
        # wakes competing readers/senders; their secondary errors must not win.
        with self._lock:
            if self.closed.is_set():
                return
            self.disconnect_reason = reason
            self.closed.set()
            for event, box in self._pending.values():
                if not event.is_set():
                    box['error'] = ChannelDisconnected(reason or 'agent control connection lost: local close')
                    event.set()
        try:
            self.sock.shutdown(2)
        except OSError:
            pass
        self.sock.close()
