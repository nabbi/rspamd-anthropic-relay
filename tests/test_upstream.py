# SPDX-License-Identifier: GPL-3.0-or-later
"""The upstream clients against a fake Messages API on loopback.

SdkParityTest is how the direct client stays current: it runs the SDK and
DirectClient against the same fake API and fails when they send a
different request (anthropic-version, path, body) or read an answer
differently. It is skipped without the SDK; CI runs it with the pinned
SDK on every push and with the newest SDK every week.
"""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from rspamd_anthropic_relay.relay import Config, to_anthropic, to_openai
from rspamd_anthropic_relay.upstream import (
    ANTHROPIC_VERSION,
    DirectClient,
    UpstreamConnectionError,
    UpstreamProtocolError,
    UpstreamStatusError,
    UpstreamTimeout,
)

try:
    import anthropic

    from rspamd_anthropic_relay.upstream import SdkClient
except ImportError:
    anthropic = None

KEY = "sk-ant-test"
REPLY = {
    "id": "msg_01",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-5-5",
    "content": [{"type": "text", "text": "0.97\nPrize lure\nscam"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 210, "output_tokens": 9},
}
REQUEST = to_anthropic(
    {"messages": [
        {"role": "system", "content": "Rate spam probability 0-1."},
        {"role": "user", "content": "Subject: You won! Ünïcödé €"},
        {"role": "user", "content": "Claim your prize."},
    ]},
    Config(token="t"),
)


class FakeAPI:
    """Records each request; answers with self.status / self.reply."""

    def __init__(self):
        self.requests = []
        self.peers = set()
        self.status = 200
        self.reply = REPLY
        self.delay = 0.0
        self.close_after = False
        api = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers["Content-Length"]))
                api.peers.add(self.client_address)
                api.requests.append((self.path, dict(self.headers), json.loads(body)))
                time.sleep(api.delay)
                data = api.reply if isinstance(api.reply, bytes) else json.dumps(api.reply).encode()
                self.send_response(api.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("request-id", "req_fake")
                self.end_headers()
                self.wfile.write(data)
                self.wfile.flush()
                if api.close_after:  # closed without saying so
                    self.close_connection = True

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request, client_address):
                pass  # a client that timed out and left: expected here

        self.server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


class DirectClientTest(unittest.TestCase):
    def setUp(self):
        self.api = FakeAPI()
        self.client = DirectClient(KEY, self.api.url, timeout=1.0)

    def tearDown(self):
        self.client.close()
        self.api.stop()

    def test_request(self):
        self.assertEqual(self.client.create(REQUEST), REPLY)
        path, headers, body = self.api.requests[0]
        self.assertEqual(path, "/v1/messages")
        self.assertEqual(headers["Authorization"], f"Bearer {KEY}")
        self.assertNotIn("x-api-key", {k.lower() for k in headers})
        self.assertEqual(headers["anthropic-version"], ANTHROPIC_VERSION)
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(body, REQUEST)

    def test_base_url_path(self):
        client = DirectClient(KEY, self.api.url + "/gateway/", timeout=1.0)
        client.create(REQUEST)
        client.close()
        self.assertEqual(self.api.requests[0][0], "/gateway/v1/messages")
        with self.assertRaises(ValueError):
            DirectClient(KEY, "ftp://example.com")

    def test_keep_alive(self):
        for _ in range(3):
            self.client.create(REQUEST)
        self.assertEqual(len(self.api.peers), 1)

    def test_closed_connection_is_not_reused(self):
        self.api.close_after = True
        for _ in range(3):
            self.assertEqual(self.client.create(REQUEST), REPLY)
        self.assertEqual(len(self.api.requests), 3)
        self.assertEqual(len(self.api.peers), 3)

    def test_retry_on_a_dead_pooled_connection(self):
        # The server closes a pooled connection just as it is reused (the
        # idle check sees nothing yet): one retry, on a new connection
        from rspamd_anthropic_relay import upstream

        self.api.close_after = True
        self.client.create(REQUEST)
        time.sleep(0.1)
        real, upstream._idle_usable = upstream._idle_usable, lambda conn: True
        try:
            self.assertEqual(self.client.create(REQUEST), REPLY)
        finally:
            upstream._idle_usable = real
        self.assertEqual(len(self.api.requests), 2)
        self.assertEqual(len(self.api.peers), 2)

    def test_status_errors(self):
        for status in (400, 429, 500, 529):
            with self.subTest(status=status):
                self.api.status = status
                self.api.reply = {"type": "error", "error": {"type": "x", "message": "no"}}
                with self.assertRaises(UpstreamStatusError) as caught:
                    self.client.create(REQUEST)
                self.assertEqual((caught.exception.status, caught.exception.request_id), (status, "req_fake"))
        self.api.status = 200
        self.api.reply = REPLY
        self.assertEqual(self.client.create(REQUEST), REPLY)  # error answers keep the connection usable

    def test_bad_answers(self):
        for reply in (b"<html>", b"[1]", b"x" * (1024 * 1024 + 10)):
            with self.subTest(reply=reply[:10]):
                self.api.reply = reply
                with self.assertRaises(UpstreamProtocolError):
                    self.client.create(REQUEST)

    def test_timeout(self):
        self.api.delay = 0.5
        client = DirectClient(KEY, self.api.url, timeout=0.2)
        started = time.monotonic()
        with self.assertRaises(UpstreamTimeout):
            client.create(REQUEST)
        self.assertLess(time.monotonic() - started, 0.45)

    def test_connection_refused(self):
        url = self.api.url
        self.api.stop()
        with self.assertRaises(UpstreamConnectionError):
            DirectClient(KEY, url, timeout=1.0).create(REQUEST)
        self.api = FakeAPI()  # for tearDown


@unittest.skipIf(anthropic is None, "the anthropic SDK is not installed (the sdk extra)")
class SdkParityTest(unittest.TestCase):
    """DirectClient must do what the SDK does."""

    def setUp(self):
        self.api = FakeAPI()
        self.direct = DirectClient(KEY, self.api.url, timeout=1.0)
        self.sdk = SdkClient(KEY, self.api.url, timeout=1.0)

    def tearDown(self):
        self.direct.close()
        self.sdk.close()
        self.api.stop()

    def test_api_version(self):
        from anthropic._constants import ANTHROPIC_API_VERSION

        self.assertEqual(
            ANTHROPIC_VERSION, ANTHROPIC_API_VERSION,
            f"anthropic {anthropic.__version__} sends anthropic-version {ANTHROPIC_API_VERSION}: "
            "check the API's changes for that version, then update ANTHROPIC_VERSION in upstream.py",
        )

    def test_same_request(self):
        self.direct.create(REQUEST)
        self.sdk.create(REQUEST)
        (dpath, dheaders, dbody), (spath, sheaders, sbody) = self.api.requests
        self.assertEqual(dpath, spath)
        self.assertEqual(dbody, sbody)
        lower = lambda h: {k.lower(): v for k, v in h.items()}  # noqa: E731
        dheaders, sheaders = lower(dheaders), lower(sheaders)
        for name in ("anthropic-version", "authorization", "x-api-key", "content-type", "accept"):
            self.assertEqual(dheaders.get(name), sheaders.get(name), name)
        # A header the SDK starts sending for this call (other than its
        # telemetry and transport ones) needs a look
        ignored = {"user-agent", "host", "content-length", "connection", "accept-encoding"}
        extra = {k for k in sheaders if k not in dheaders and k not in ignored and not k.startswith("x-stainless-")}
        self.assertEqual(extra, set(), "headers the SDK sends and the direct client does not")

    def test_same_reply(self):
        a, b = to_openai(self.direct.create(REQUEST)), to_openai(self.sdk.create(REQUEST))
        a.pop("created"), b.pop("created")
        self.assertEqual(a, b)

    def test_same_errors(self):
        for status in (429, 500, 529):
            with self.subTest(status=status):
                self.api.status = status
                self.api.reply = {"type": "error", "error": {"type": "x", "message": "no"}}
                for client in (self.direct, self.sdk):
                    with self.assertRaises(UpstreamStatusError) as caught:
                        client.create(REQUEST)
                    self.assertEqual((caught.exception.status, caught.exception.request_id), (status, "req_fake"))
        self.api.status, self.api.reply, self.api.delay = 200, REPLY, 0.5
        for client in (DirectClient(KEY, self.api.url, timeout=0.2), SdkClient(KEY, self.api.url, timeout=0.2)):
            with self.assertRaises(UpstreamTimeout):
                client.create(REQUEST)
            client.close()


if __name__ == "__main__":
    unittest.main()
