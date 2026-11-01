"""A small HTTP relay: OpenAI-style chat completions in, Anthropic's
Messages API out.

rspamd's gpt module speaks only the OpenAI Chat Completions format. This
relay accepts that request on a local address, sends it to Claude through
the official Anthropic SDK, and answers in the shape rspamd parses
(choices[0].message.content, finish_reason, usage). It is deliberately
narrow: one endpoint, text only, no streaming, no tools.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import logging.handlers
import os
import signal
import socket
import sys
import threading
import time
import zlib
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import anthropic

from . import __version__

log = logging.getLogger("rspamd-anthropic-relay")

CHAT_PATHS = ("/v1/chat/completions", "/chat/completions")
HEALTH_PATHS = ("/healthz", "/health")

# Anthropic stop_reason -> OpenAI finish_reason. rspamd treats "length"
# specially (a truncated reply) and anything with empty content as a failure.
FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "refusal": "content_filter",
}


class RequestError(Exception):
    """A client error: answered with an OpenAI-style error body."""

    def __init__(self, status: HTTPStatus, message: str, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 11338
    token: str = ""
    model: str = "claude-haiku-5-5"
    max_tokens: int = 512
    thinking: str = "disabled"  # disabled | adaptive | omit
    effort: str = "low"  # low | medium | high | xhigh | max | "" (omit)
    timeout: float = 4.5
    max_body: int = 1024 * 1024
    max_concurrent: int = 4


def _read_secret(env: dict[str, str], name: str) -> str:
    """NAME_FILE (a file holding the value) wins over NAME."""
    path = env.get(name + "_FILE")
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    return env.get(name, "").strip()


def load_env_file(path: str) -> dict[str, str]:
    """KEY=value lines; blank lines and # comments skipped. A value may be
    wrapped in single or double quotes, which are removed."""
    values: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            key, sep, value = line.partition("=")
            key = key.strip()
            if line.startswith("export "):
                key = key[len("export "):].strip()
            if not sep or not key.replace("_", "").isalnum():
                raise ValueError(f"{path}:{n}: not KEY=value")
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            values[key] = value
    return values


def _split_listen(value: str) -> tuple[str, int]:
    host, sep, port = value.rpartition(":")
    if not sep or not port.isdigit():
        raise ValueError(f"RELAY_LISTEN must be host:port, not {value!r}")
    return host.strip("[]") or "127.0.0.1", int(port)


def config_from_env(env: dict[str, str] | None = None) -> Config:
    env = dict(os.environ if env is None else env)
    host, port = _split_listen(env.get("RELAY_LISTEN", "127.0.0.1:11338"))
    cfg = Config(
        host=host,
        port=port,
        token=_read_secret(env, "RELAY_TOKEN"),
        model=env.get("RELAY_MODEL", Config.model),
        max_tokens=int(env.get("RELAY_MAX_TOKENS", Config.max_tokens)),
        thinking=env.get("RELAY_THINKING", Config.thinking),
        effort=env.get("RELAY_EFFORT", Config.effort),
        timeout=float(env.get("RELAY_TIMEOUT", Config.timeout)),
        max_body=int(env.get("RELAY_MAX_BODY", Config.max_body)),
        max_concurrent=int(env.get("RELAY_MAX_CONCURRENT", Config.max_concurrent)),
    )
    if not cfg.token:
        raise ValueError("RELAY_TOKEN (or RELAY_TOKEN_FILE) is required")
    if cfg.thinking not in ("disabled", "adaptive", "omit"):
        raise ValueError("RELAY_THINKING must be disabled, adaptive or omit")
    if cfg.effort not in ("", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("RELAY_EFFORT must be low, medium, high, xhigh, max or empty")
    if cfg.max_tokens < 1:
        raise ValueError("RELAY_MAX_TOKENS must be at least 1")
    return cfg


def _text_of(content: Any) -> str:
    """OpenAI message content: a string, or a list of text parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                texts.append(str(part.get("text", "")))
            else:
                raise RequestError(HTTPStatus.BAD_REQUEST, "only text content parts are supported")
        return "\n".join(texts)
    raise RequestError(HTTPStatus.BAD_REQUEST, "message content must be a string or a list of text parts")


def to_anthropic(request: Any, cfg: Config) -> dict[str, Any]:
    """Build Messages API arguments from a chat completions request.

    System and developer messages are joined into the system prompt.
    Consecutive messages of the same role are joined with newlines (rspamd
    sends subject, sender, link domains and text as separate user
    messages). The model is always the configured one.
    """
    if not isinstance(request, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "request body must be a JSON object")
    if request.get("stream"):
        raise RequestError(HTTPStatus.BAD_REQUEST, "streaming is not supported")
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise RequestError(HTTPStatus.BAD_REQUEST, "messages must be a non-empty list")

    system: list[str] = []
    turns: list[dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            raise RequestError(HTTPStatus.BAD_REQUEST, "each message must be an object")
        role = m.get("role")
        text = _text_of(m.get("content"))
        if role in ("system", "developer"):
            if text.strip():
                system.append(text)
        elif role in ("user", "assistant"):
            if not text.strip():
                continue
            if turns and turns[-1]["role"] == role:
                turns[-1]["content"] += "\n" + text
            else:
                turns.append({"role": role, "content": text})
        else:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"unsupported message role: {role!r}")

    if not turns or turns[0]["role"] != "user":
        raise RequestError(HTTPStatus.BAD_REQUEST, "the first non-system message must be from the user")
    if turns[-1]["role"] != "user":
        raise RequestError(HTTPStatus.BAD_REQUEST, "the last message must be from the user")

    asked = request.get("max_completion_tokens") or request.get("max_tokens") or cfg.max_tokens
    try:
        max_tokens = max(1, min(int(asked), cfg.max_tokens))
    except (TypeError, ValueError):
        raise RequestError(HTTPStatus.BAD_REQUEST, "max_tokens must be an integer") from None

    kwargs: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": max_tokens,
        "messages": turns,
    }
    if system:
        kwargs["system"] = "\n\n".join(system)
    if cfg.thinking != "omit":
        kwargs["thinking"] = {"type": cfg.thinking}
    if cfg.effort:
        kwargs["output_config"] = {"effort": cfg.effort}
    return kwargs


def to_openai(message: Any) -> dict[str, Any]:
    """A chat completions response from an Anthropic Message."""
    text = "".join(b.text for b in message.content if getattr(b, "type", None) == "text")
    usage = message.usage
    prompt = getattr(usage, "input_tokens", 0) or 0
    completion = getattr(usage, "output_tokens", 0) or 0
    return {
        "id": message.id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": message.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": FINISH_REASONS.get(message.stop_reason or "", "stop"),
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        },
    }


def _decompress(body: bytes, limit: int) -> bytes:
    """gzip or zlib (rspamd gzips its requests), at most limit bytes."""
    d = zlib.decompressobj(wbits=47)  # 32 + 15: gzip or zlib header
    out = d.decompress(body, limit + 1)
    if len(out) > limit or d.unconsumed_tail:
        raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "decompressed body too large")
    return out


class Relay:
    """The server's shared state: config, the Anthropic client, a limit."""

    def __init__(self, cfg: Config, client: Any):
        self.cfg = cfg
        self.client = client
        self.slots = threading.BoundedSemaphore(cfg.max_concurrent)

    def complete(self, request: Any) -> tuple[HTTPStatus, dict[str, Any], dict[str, Any]]:
        """Returns (status, response body, fields for the log line)."""
        kwargs = to_anthropic(request, self.cfg)
        info: dict[str, Any] = {"model": self.cfg.model, "asked": request.get("model")}
        try:
            message = self.client.messages.create(**kwargs)
        except anthropic.APITimeoutError:
            raise RequestError(HTTPStatus.GATEWAY_TIMEOUT, "upstream timeout", "timeout") from None
        except anthropic.APIConnectionError:
            raise RequestError(HTTPStatus.BAD_GATEWAY, "upstream connection error", "api_connection_error") from None
        except anthropic.RateLimitError as e:
            raise RequestError(HTTPStatus.TOO_MANY_REQUESTS, f"upstream rate limit ({e.request_id})", "rate_limit_error") from None
        except anthropic.APIStatusError as e:
            raise RequestError(HTTPStatus.BAD_GATEWAY, f"upstream status {e.status_code} ({e.request_id})", "api_error") from None
        info.update(
            id=message.id,
            stop=message.stop_reason,
            input=getattr(message.usage, "input_tokens", None),
            output=getattr(message.usage, "output_tokens", None),
        )
        return HTTPStatus.OK, to_openai(message), info


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"rspamd-anthropic-relay/{__version__}"
    relay: Relay  # set on the subclass made by make_server

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        log.debug("%s %s", self.address_string(), format % args)

    def _send(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        data = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, err: RequestError) -> None:
        self._send(err.status, {"error": {"message": err.message, "type": err.kind}})

    def do_GET(self) -> None:  # noqa: N802
        if self.path in HEALTH_PATHS:
            self._send(HTTPStatus.OK, {"status": "ok"})
        else:
            self._error(RequestError(HTTPStatus.NOT_FOUND, "not found", "not_found"))

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        scheme, _, token = header.partition(" ")
        return scheme.lower() == "bearer" and hmac.compare_digest(
            token.strip().encode(), self.relay.cfg.token.encode()
        )

    def _body(self) -> Any:
        cfg = self.relay.cfg
        if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            self.close_connection = True
            raise RequestError(HTTPStatus.LENGTH_REQUIRED, "chunked bodies are not supported")
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            raise RequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length required") from None
        if length < 0 or length > cfg.max_body:
            self.close_connection = True
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "body too large")
        body = self.rfile.read(length)
        encoding = self.headers.get("Content-Encoding", "identity").strip().lower()
        if encoding in ("gzip", "deflate", "x-gzip"):
            try:
                body = _decompress(body, cfg.max_body)
            except zlib.error:
                raise RequestError(HTTPStatus.BAD_REQUEST, "cannot decompress body") from None
        elif encoding not in ("", "identity"):
            raise RequestError(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, f"unsupported Content-Encoding {encoding}")
        try:
            return json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise RequestError(HTTPStatus.BAD_REQUEST, "body is not valid JSON") from None

    def do_POST(self) -> None:  # noqa: N802
        started = time.monotonic()
        status: HTTPStatus = HTTPStatus.INTERNAL_SERVER_ERROR
        info: dict[str, Any] = {}
        acquired = False
        try:
            if self.path not in CHAT_PATHS:
                raise RequestError(HTTPStatus.NOT_FOUND, "not found", "not_found")
            # Read the body before any refusal, so a kept-alive connection
            # stays in step
            request = self._body()
            if not self._authorized():
                raise RequestError(HTTPStatus.UNAUTHORIZED, "invalid token", "authentication_error")
            acquired = self.relay.slots.acquire(blocking=False)
            if not acquired:
                raise RequestError(HTTPStatus.SERVICE_UNAVAILABLE, "too many requests in flight", "overloaded")
            status, body, info = self.relay.complete(request)
            self._send(status, body)
        except RequestError as err:
            status = err.status
            info.setdefault("error", err.message)
            self._error(err)
        except Exception:
            log.exception("unexpected error")
            status = HTTPStatus.INTERNAL_SERVER_ERROR
            self._error(RequestError(status, "internal error", "internal_error"))
        finally:
            if acquired:
                self.relay.slots.release()
            fields = " ".join(f"{k}={v}" for k, v in info.items() if v is not None)
            log.info(
                "%s %d %dms %s",
                self.path,
                status,
                (time.monotonic() - started) * 1000,
                fields,
            )


def make_server(cfg: Config, client: Any) -> ThreadingHTTPServer:
    relay = Relay(cfg, client)
    handler = type("BoundHandler", (Handler,), {"relay": relay})
    server_class = ThreadingHTTPServer
    if ":" in cfg.host:
        server_class = type("ThreadingHTTPServer6", (ThreadingHTTPServer,), {"address_family": socket.AF_INET6})
    server = server_class((cfg.host, cfg.port), handler)
    server.daemon_threads = True
    return server


def _setup_logging(target: str, facility: str, level: str) -> None:
    if target == "syslog":
        handler: logging.Handler = logging.handlers.SysLogHandler(
            address="/dev/log",
            facility=logging.handlers.SysLogHandler.facility_names[facility],
        )
        handler.setFormatter(logging.Formatter("rspamd-anthropic-relay[%(process)d]: %(message)s"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(level.upper())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rspamd-anthropic-relay",
        description="Relay OpenAI-style chat completions (rspamd's gpt module) to Anthropic's Messages API.",
        epilog="Configuration is read from the environment: see README.md.",
    )
    parser.add_argument(
        "--env-file",
        default=os.environ.get("RELAY_ENV_FILE"),
        help="KEY=value settings, applied over the environment",
    )
    parser.add_argument("--log", choices=("stderr", "syslog"), default=os.environ.get("RELAY_LOG", "stderr"))
    parser.add_argument("--syslog-facility", default=os.environ.get("RELAY_SYSLOG_FACILITY", "mail"))
    parser.add_argument("--log-level", default=os.environ.get("RELAY_LOG_LEVEL", "info"))
    parser.add_argument("--version", action="version", version=__version__)
    args = parser.parse_args(argv)
    _setup_logging(args.log, args.syslog_facility, args.log_level)

    try:
        env = dict(os.environ)
        if args.env_file:
            env.update(load_env_file(args.env_file))
        cfg = config_from_env(env)
        api_key = _read_secret(env, "ANTHROPIC_API_KEY")
    except (ValueError, OSError) as e:
        log.error("configuration: %s", e)
        return 2
    if not api_key:
        log.error("configuration: ANTHROPIC_API_KEY (or ANTHROPIC_API_KEY_FILE) is required")
        return 2

    # The environment and --env-file both count, for the SDK's settings too
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=env.get("ANTHROPIC_BASE_URL") or None,
        timeout=cfg.timeout,
        max_retries=0,
    )
    server = make_server(cfg, client)

    def stop(signum: int, _frame: Any) -> None:
        log.info("signal %d, stopping", signum)
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info(
        "listening on %s:%d, model %s, thinking %s, effort %s, max_tokens %d, timeout %.1fs",
        cfg.host, cfg.port, cfg.model, cfg.thinking, cfg.effort or "default", cfg.max_tokens, cfg.timeout,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0
