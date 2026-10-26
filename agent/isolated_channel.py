"""Correlated duplex RPC; readers never block behind an approval callback."""
import threading
import uuid

import time

from agent.isolated_transport import receive_message, send_message, TRANSFER_TIMEOUT


class Channel:
    def __init__(self, sock, generation, handler, on_disconnect=None):
        self.sock, self.generation, self.handler = sock, generation, handler
        self.on_disconnect = on_disconnect
        self.closed = threading.Event()
        self._send_lock = threading.Lock()
        self._lock = threading.Lock()
        self._pending = {}
        self._slots = threading.BoundedSemaphore(8)
        self._thread = threading.Thread(target=self._read, daemon=True, name='agent-control')
        self._thread.start()

    def _send(self, message):
        deadline = time.monotonic() + TRANSFER_TIMEOUT
        if not self._send_lock.acquire(timeout=TRANSFER_TIMEOUT):
            self.close()
            raise TimeoutError('agent control send queue timed out')
        try:
            if self.closed.is_set():
                raise EOFError('agent control closed')
            send_message(self.sock, dict(message, v=1, generation=self.generation), deadline)
        except (OSError, EOFError):
            self.close()
            raise
        finally:
            self._send_lock.release()

    def call(self, op, payload=None, timeout=30):
        key = uuid.uuid4().hex
        event, box = threading.Event(), {}
        with self._lock:
            if len(self._pending) >= 8:
                raise RuntimeError('agent control request limit reached')
            self._pending[key] = (event, box)
        try:
            self._send({'id': key, 'op': op, 'payload': payload})
            if not event.wait(timeout):
                raise TimeoutError('agent control acknowledgement timed out')
            if 'error' in box:
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
        except (OSError, EOFError, ValueError, TypeError, RecursionError):
            self.close()  # Release all waiters if even the terminal reply fails.
        finally:
            self._slots.release()

    def _read(self):
        try:
            while not self.closed.is_set():
                message = receive_message(self.sock, self.generation)
                if 'reply' in message:
                    with self._lock:
                        pending = self._pending.get(message['reply'])
                        if pending is not None:
                            event, box = pending
                            box.update(message)
                            event.set()
                else:
                    if not self._slots.acquire(blocking=False):
                        raise ValueError('agent control concurrency exceeded')
                    threading.Thread(target=self._dispatch, args=(message,), daemon=True).start()
        except (OSError, EOFError, ValueError, KeyError, TypeError, RecursionError):
            pass
        finally:
            self.close()
            if self.on_disconnect:
                self.on_disconnect()

    def close(self):
        self.closed.set()
        try:
            self.sock.shutdown(2)
        except OSError:
            pass
        self.sock.close()
        with self._lock:
            for event, box in self._pending.values():
                if not event.is_set():
                    box['error'] = 'agent control connection lost'
                    event.set()
