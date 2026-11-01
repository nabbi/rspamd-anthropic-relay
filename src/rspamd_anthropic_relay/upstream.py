# SPDX-License-Identifier: GPL-3.0-or-later
"""The Messages API call, two ways.

DirectClient (RELAY_BACKEND=direct, the default) is one POST /v1/messages
with the standard library: no dependencies, ~15 MiB of memory instead of
the SDK's ~80. SdkClient (RELAY_BACKEND=sdk) does the same through the
official Anthropic SDK, installed with the "sdk" extra.

Both take the Messages API arguments as a dict, return the response as
the API's JSON (a dict), and raise the UpstreamError subclasses below.
tests/test_upstream.py runs both against the same fake API and checks
that they send the same request and read the same answer; ANTHROPIC_VERSION
must match what the SDK sends, and that test says when it no longer does.
"""

from __future__ import annotations

import http.client
import json
import select
import ssl
import threading
import time
import urllib.parse
from typing import Any

from . import __version__

# The anthropic-version header. Keep it equal to the SDK's
# anthropic._constants.ANTHROPIC_API_VERSION (tests/test_upstream.py).
ANTHROPIC_VERSION = "2026-10-21"
DEFAULT_BASE_URL = "https://api.anthropic.com"
USER_AGENT = f"rspamd-anthropic-relay/{__version__}"
# A reply is a few hundred tokens; anything near this is not one
MAX_RESPONSE = 1024 * 1024
# Idle upstream connections are reused for this long, then reconnected
POOL_IDLE = 30.0


class UpstreamError(Exception):
    """The Messages API call failed."""


class UpstreamTimeout(UpstreamError):
    pass


class UpstreamConnectionError(UpstreamError):
    pass


class UpstreamProtocolError(UpstreamError):
    """An answer that is not a Messages API response."""


class UpstreamStatusError(UpstreamError):
    def __init__(self, status: int, request_id: str | None):
        super().__init__(f"status {status}")
        self.status = status
        self.request_id = request_id


def _idle_usable(conn: http.client.HTTPConnection) -> bool:
    """An idle connection is usable while the server has sent nothing on
    it: readable means closed (or junk), either way not reusable."""
    sock = conn.sock
    if sock is None:
        return False
    try:
        if isinstance(sock, ssl.SSLSocket) and sock.pending():
            return False
        readable, _, _ = select.select([sock], [], [], 0)
    except (OSError, ValueError):
        return False
    return not readable


class DirectClient:
    """POST /v1/messages over http.client, with a small keep-alive pool
    (at most pool_size idle connections). Thread-safe: a connection is
    used by one request at a time. No retries, except one on a fresh
    connection when a reused one turns out to be closed before it
    answered. timeout bounds the whole call."""

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 4.5, pool_size: int = 4):
        url = urllib.parse.urlsplit(base_url or DEFAULT_BASE_URL)
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ValueError(f"ANTHROPIC_BASE_URL must be an http(s) URL, not {base_url!r}")
        self.https = url.scheme == "https"
        self.host = url.hostname
        self.port = url.port or (443 if self.https else 80)
        self.path = url.path.rstrip("/") + "/v1/messages"
        self.timeout = timeout
        self.pool_size = pool_size
        self._ssl = ssl.create_default_context() if self.https else None
        self._idle: list[tuple[http.client.HTTPConnection, float]] = []
        self._lock = threading.Lock()
        self._headers = {
            "Host": url.netloc,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
            "anthropic-version": ANTHROPIC_VERSION,
            # As the SDK sends it since 2.0 (with this anthropic-version);
            # before, it was x-api-key
            "Authorization": f"Bearer {api_key}",
        }

    def _connection(self, timeout: float, fresh: bool) -> tuple[http.client.HTTPConnection, bool]:
        now = time.monotonic()
        with self._lock:
            while self._idle and not fresh:
                conn, since = self._idle.pop()
                if now - since < POOL_IDLE and _idle_usable(conn):
                    return conn, True
                conn.close()
        if self.https:
            return http.client.HTTPSConnection(self.host, self.port, timeout=timeout, context=self._ssl), False
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout), False

    def _release(self, conn: http.client.HTTPConnection) -> None:
        with self._lock:
            if len(self._idle) < self.pool_size:
                self._idle.append((conn, time.monotonic()))
                return
        conn.close()

    def close(self) -> None:
        with self._lock:
            idle, self._idle = self._idle, []
        for conn, _ in idle:
            conn.close()

    def create(self, body: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
        headers = dict(self._headers, **{"Content-Length": str(len(data))})
        deadline = time.monotonic() + self.timeout
        for attempt in (1, 2):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise UpstreamTimeout("timeout")
            conn, reused = self._connection(remaining, fresh=attempt > 1)
            try:
                if conn.sock is not None:
                    conn.sock.settimeout(remaining)
                conn.request("POST", self.path, body=data, headers=headers)
                sock = conn.sock  # getresponse() drops it when the server closes
                resp = conn.getresponse()
                if sock is not None:
                    sock.settimeout(max(deadline - time.monotonic(), 0.001))
                raw = resp.read(MAX_RESPONSE + 1)
            except TimeoutError:
                conn.close()
                raise UpstreamTimeout("timeout") from None
            except (ConnectionResetError, BrokenPipeError):
                # http.client.RemoteDisconnected is a ConnectionResetError:
                # a pooled connection the server closed as it was reused
                conn.close()
                if reused and attempt == 1:
                    continue
                raise UpstreamConnectionError("connection lost") from None
            except (OSError, http.client.HTTPException) as e:
                conn.close()
                raise UpstreamConnectionError(type(e).__name__) from None
            break
        if resp.isclosed() and not resp.will_close and len(raw) <= MAX_RESPONSE:
            self._release(conn)
        else:
            conn.close()
        if len(raw) > MAX_RESPONSE:
            raise UpstreamProtocolError("response too large")
        if resp.status != 200:
            raise UpstreamStatusError(resp.status, resp.getheader("request-id"))
        try:
            message = json.loads(raw)
        except (ValueError, RecursionError):
            raise UpstreamProtocolError("response is not JSON") from None
        if not isinstance(message, dict):
            raise UpstreamProtocolError("response is not an object")
        return message


class SdkClient:
    """The same call through the Anthropic SDK (pip install
    'rspamd-anthropic-relay[sdk]'). It honours the SDK's own environment,
    HTTPS_PROXY among it, which DirectClient does not."""

    def __init__(self, api_key: str, base_url: str | None = None, timeout: float = 4.5, pool_size: int = 4):
        import anthropic  # only for this backend: ~65 MiB

        self._anthropic = anthropic
        limits = type(anthropic.DEFAULT_CONNECTION_LIMITS)(
            max_connections=pool_size, max_keepalive_connections=pool_size
        )
        self.client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            max_retries=0,
            http_client=anthropic.DefaultHttpxClient(limits=limits, timeout=timeout),
        )

    def close(self) -> None:
        self.client.close()

    def create(self, body: dict[str, Any]) -> dict[str, Any]:
        a = self._anthropic
        try:
            message = self.client.messages.create(**body)
        except a.APITimeoutError:
            raise UpstreamTimeout("timeout") from None
        except a.APIConnectionError:
            raise UpstreamConnectionError("connection error") from None
        except a.APIStatusError as e:
            raise UpstreamStatusError(e.status_code, e.request_id) from None
        except a.APIResponseValidationError:
            raise UpstreamProtocolError("invalid response") from None
        return message.model_dump(mode="json")


BACKENDS = {"direct": DirectClient, "sdk": SdkClient}


def make_client(backend: str, api_key: str, base_url: str | None, timeout: float, pool_size: int) -> Any:
    """Raises ImportError for "sdk" without the SDK installed."""
    return BACKENDS[backend](api_key, base_url, timeout, pool_size)
