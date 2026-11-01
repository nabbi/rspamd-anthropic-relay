# SPDX-License-Identifier: GPL-3.0-or-later
import gzip
import http.client
import json
import threading
import unittest
from types import SimpleNamespace

import anthropic
import httpx2

from rspamd_anthropic_relay.relay import (
    Config,
    RequestError,
    config_from_env,
    load_env_file,
    make_server,
    to_anthropic,
    to_openai,
)

TOKEN = "test-token"

# What rspamd's gpt module sends (gpt.lua, openai type, plain replies)
RSPAMD_REQUEST = {
    "stream": False,
    "model": "claude-haiku-5-5",
    "max_tokens": 1000,
    "messages": [
        {"role": "system", "content": "Analyze this email as a spam detector."},
        {"role": "user", "content": "Subject: Your invoice"},
        {"role": "user", "content": "billing@example.com"},
        {"role": "user", "content": "example.com"},
        {"role": "user", "content": "Please find your invoice attached."},
    ],
}


def message(text="0.92\nInvoice lure from an unknown sender\nphishing", stop="end_turn"):
    return SimpleNamespace(
        id="msg_test",
        model="claude-haiku-5-5",
        stop_reason=stop,
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=120, output_tokens=14),
    )


class FakeClient:
    def __init__(self, result=None, error=None):
        self.calls = []
        self.result = result or message()
        self.error = error
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


class ConversionTest(unittest.TestCase):
    cfg = Config(token=TOKEN)

    def test_rspamd_request(self):
        kw = to_anthropic(RSPAMD_REQUEST, self.cfg)
        self.assertEqual(kw["model"], "claude-haiku-5-5")
        self.assertEqual(kw["system"], "Analyze this email as a spam detector.")
        self.assertEqual(len(kw["messages"]), 1)
        self.assertEqual(
            kw["messages"][0]["content"],
            "Subject: Your invoice\nbilling@example.com\nexample.com\nPlease find your invoice attached.",
        )
        self.assertEqual(kw["max_tokens"], 512)  # capped by the relay
        self.assertEqual(kw["thinking"], {"type": "disabled"})
        self.assertEqual(kw["output_config"], {"effort": "low"})
        for dropped in ("temperature", "top_p", "response_format", "stream"):
            self.assertNotIn(dropped, kw)

    def test_model_is_the_configured_one(self):
        kw = to_anthropic(dict(RSPAMD_REQUEST, model="gpt-5-mini"), self.cfg)
        self.assertEqual(kw["model"], "claude-haiku-5-5")

    def test_thinking_and_effort_omitted(self):
        cfg = Config(token=TOKEN, thinking="omit", effort="")
        kw = to_anthropic(RSPAMD_REQUEST, cfg)
        self.assertNotIn("thinking", kw)
        self.assertNotIn("output_config", kw)

    def test_text_parts_and_empty_messages(self):
        req = {"messages": [
            {"role": "user", "content": ""},
            {"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]},
        ]}
        kw = to_anthropic(req, self.cfg)
        self.assertEqual(kw["messages"], [{"role": "user", "content": "a\nb"}])
        self.assertNotIn("system", kw)

    def test_rejections(self):
        bad = [
            [],
            {"messages": []},
            {"messages": [{"role": "system", "content": "only system"}]},
            {"messages": [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]},
            {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "prefill"}]},
            {"messages": [{"role": "tool", "content": "x"}]},
            {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
            {"messages": [{"role": "user", "content": "x"}], "stream": True},
            {"messages": [{"role": "user", "content": "x"}], "max_tokens": "many"},
        ]
        for req in bad:
            with self.subTest(req=req), self.assertRaises(RequestError):
                to_anthropic(req, self.cfg)

    def test_reply(self):
        out = to_openai(message())
        self.assertEqual(out["choices"][0]["message"]["content"], "0.92\nInvoice lure from an unknown sender\nphishing")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")
        self.assertEqual(out["usage"], {"prompt_tokens": 120, "completion_tokens": 14, "total_tokens": 134})
        self.assertEqual(to_openai(message(stop="max_tokens"))["choices"][0]["finish_reason"], "length")
        refused = to_openai(message(text="", stop="refusal"))
        self.assertEqual(refused["choices"][0]["finish_reason"], "content_filter")
        self.assertEqual(refused["choices"][0]["message"]["content"], "")

    def test_config(self):
        cfg = config_from_env({"RELAY_TOKEN": "t", "RELAY_LISTEN": "[::1]:9000", "RELAY_EFFORT": ""})
        self.assertEqual((cfg.host, cfg.port, cfg.effort), ("::1", 9000, ""))
        for env in ({}, {"RELAY_TOKEN": "t", "RELAY_THINKING": "on"}, {"RELAY_TOKEN": "t", "RELAY_LISTEN": "nope"}):
            with self.subTest(env=env), self.assertRaises(ValueError):
                config_from_env(env)


class EnvFileTest(unittest.TestCase):
    def test_env_file(self):
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False) as f:
            f.write("# comment\n\nRELAY_TOKEN='abc def'\nexport RELAY_MODEL=\"claude-haiku-5-5\"\nRELAY_EFFORT=\n")
        values = load_env_file(f.name)
        self.assertEqual(values, {"RELAY_TOKEN": "abc def", "RELAY_MODEL": "claude-haiku-5-5", "RELAY_EFFORT": ""})
        with open(f.name, "a") as g:
            g.write("not a setting\n")
        with self.assertRaises(ValueError):
            load_env_file(f.name)


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.server = make_server(Config(port=0, token=TOKEN, max_body=4096), self.client)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, body, token=TOKEN, gzipped=False, path="/v1/chat/completions", conn=None):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"}
        if gzipped:
            data = gzip.compress(data)
            headers["Content-Encoding"] = "gzip"
        if token is not None:
            headers["Authorization"] = f"Bearer {token}"
        conn = conn or http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("POST", path, body=data, headers=headers)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())

    def test_gzipped_rspamd_request_kept_alive(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        for _ in range(2):  # rspamd keeps connections alive
            status, body = self.post(RSPAMD_REQUEST, gzipped=True, conn=conn)
            self.assertEqual(status, 200)
            self.assertTrue(body["choices"][0]["message"]["content"].startswith("0.92"))
        self.assertEqual(len(self.client.calls), 2)

    def test_auth(self):
        self.assertEqual(self.post(RSPAMD_REQUEST, token="wrong")[0], 401)
        self.assertEqual(self.post(RSPAMD_REQUEST, token=None)[0], 401)
        self.assertEqual(self.client.calls, [])

    def test_bad_requests(self):
        self.assertEqual(self.post(b"not json")[0], 400)
        self.assertEqual(self.post(RSPAMD_REQUEST, path="/v1/embeddings")[0], 404)
        self.assertEqual(self.post({"messages": [{"role": "user", "content": "x" * 5000}]})[0], 413)
        bomb = {"messages": [{"role": "user", "content": "x" * 100000}]}
        self.assertEqual(self.post(bomb, gzipped=True)[0], 413)

    def test_health(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        conn.request("GET", "/healthz")
        self.assertEqual(conn.getresponse().status, 200)

    def test_upstream_errors(self):
        request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
        cases = [
            (anthropic.APITimeoutError(request=request), 504),
            (anthropic.APIConnectionError(request=request), 502),
            (anthropic.InternalServerError("boom", response=httpx2.Response(500, request=request), body=None), 502),
            (anthropic.RateLimitError("slow", response=httpx2.Response(429, request=request), body=None), 429),
        ]
        for error, status in cases:
            with self.subTest(error=type(error).__name__):
                self.client.error = error
                got, body = self.post(RSPAMD_REQUEST)
                self.assertEqual(got, status)
                self.assertIn("error", body)


if __name__ == "__main__":
    unittest.main()
