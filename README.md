# rspamd-anthropic-relay

Use Claude for [rspamd](https://rspamd.com)'s LLM check, through Anthropic's
own Messages API.

rspamd's `gpt` module speaks only the OpenAI Chat Completions format (its
types are `openai` and `ollama`). This relay listens on a local address,
takes the module's request, sends it to Claude with the official
[Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python),
and answers in the shape the module parses. rspamd itself stays stock:
`type = "openai"` with `url` pointing at the relay.

```
rspamd (gpt module) --HTTP, loopback--> relay --HTTPS--> api.anthropic.com/v1/messages
      OpenAI chat completions                     Anthropic Messages API (SDK)
```

Why not Anthropic's OpenAI-compatible endpoint? It works for trying
things out, but Anthropic documents it as "primarily intended to test and
compare model capabilities, and is not considered a long-term or
production-ready solution". The relay uses the supported API instead and
is small enough to read in one sitting (`src/rspamd_anthropic_relay/relay.py`).

## What it does

- `POST /v1/chat/completions` (and `/chat/completions`), `GET /healthz`.
- System and developer messages become the system prompt; consecutive
  messages of one role are joined with newlines (rspamd sends the subject,
  sender, link domains and body as separate user messages).
- The model is always the relay's configured `RELAY_MODEL`, whatever the
  request names. `max_tokens` / `max_completion_tokens` are capped at
  `RELAY_MAX_TOKENS`. `temperature`, `top_p`, `response_format` and other
  OpenAI fields are dropped (current Claude models reject sampling
  parameters).
- Thinking is off and effort `low` by default: a two- or three-line spam
  verdict needs neither, and rspamd waits on the answer.
- Replies: the text, `finish_reason` (`stop`, `length`, or `content_filter`
  for a refusal, with empty content, which rspamd treats as no verdict) and
  token `usage`.
- Requests may be gzip-compressed (rspamd compresses them). Bodies are
  limited in size, compressed and uncompressed.
- Errors come back as OpenAI-style error bodies: 401 for a wrong token, 429
  when Anthropic rate-limits, 502 for other upstream errors, 503 when
  `RELAY_MAX_CONCURRENT` requests are already in flight, 504 on timeout.
  rspamd adds no symbol for a failed request, so mail keeps flowing.
- One log line per request: path, status, time, model, Anthropic's message
  id, stop reason and token counts. Never the message content, the token
  or the key.

Not supported: streaming, tools, images, multiple choices (`n`). The
relay retries nothing: rspamd's time per message is short, and a missed
verdict costs little.

## Install

Python 3.11 or later (the pinned httpx2 stack needs it).

```sh
python3 -m venv /opt/rspamd-anthropic-relay/venv
/opt/rspamd-anthropic-relay/venv/bin/pip install --require-hashes -r requirements.txt
/opt/rspamd-anthropic-relay/venv/bin/pip install --no-deps .
```

`requirements.txt` pins every dependency with hashes (`pip-compile
--generate-hashes`). Without the package install, run it from a checkout:
`PYTHONPATH=src python -m rspamd_anthropic_relay`.

Create an unprivileged account for it (the service files use
`rspamd-relay`) and its settings file, mode 0600, owned by that account:
start from [`contrib/env.example`](contrib/env.example).

Service files: [`contrib/openrc`](contrib/openrc) (init script and conf.d)
and [`contrib/systemd`](contrib/systemd).

## Configuration

Environment variables, or `KEY=value` lines in the file given with
`--env-file` (or `RELAY_ENV_FILE`), which override the environment.

| Variable | Default | |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_API_KEY_FILE` | required | The Anthropic API key, or a file holding it |
| `RELAY_TOKEN` / `RELAY_TOKEN_FILE` | required | What rspamd must send as `Authorization: Bearer` (its `api_key`) |
| `ANTHROPIC_BASE_URL` | the SDK's | Another Messages API endpoint (a test server, a gateway) |
| `RELAY_LISTEN` | `127.0.0.1:11338` | `host:port`; `[::1]:11338` for IPv6 |
| `RELAY_MODEL` | `claude-haiku-5-5` | The Claude model every request uses |
| `RELAY_MAX_TOKENS` | `512` | Cap on the reply's tokens |
| `RELAY_THINKING` | `disabled` | `disabled`, `adaptive`, or `omit` (the model's default) |
| `RELAY_EFFORT` | `low` | `low` … `max`, or empty for the model's default |
| `RELAY_TIMEOUT` | `4.5` | Seconds for the Anthropic call; keep it under rspamd's `timeout` |
| `RELAY_MAX_CONCURRENT` | `4` | Requests in flight before answering 503 |
| `RELAY_MAX_BODY` | `1048576` | Request size limit in bytes, before and after decompression |
| `RELAY_LOG` | `stderr` | `stderr` or `syslog` (also `--log`) |
| `RELAY_SYSLOG_FACILITY` | `mail` | |
| `RELAY_LOG_LEVEL` | `info` | |

Keep it on loopback. The token stops other local users from spending your
API credit through it; the relay has no TLS of its own.

Check a model's settings against Anthropic's documentation before
changing them: for example, some models accept `disabled` thinking only
at some effort levels, and a request the API rejects becomes a 502 here.

## rspamd

`/etc/rspamd/local.d/gpt.conf`, as in [`contrib/rspamd/gpt.conf`](contrib/rspamd/gpt.conf):

```
type = "openai";
url = "http://127.0.0.1:11338/v1/chat/completions";
api_key = "<RELAY_TOKEN>";
model = "claude-haiku-5-5";
model_parameters = { "claude-haiku-5-5" = { max_tokens = 1000 } };
timeout = 5s;
autolearn = false;
enabled = true;
```

Things to know about the module (rspamd 3.14):

- It skips mail that is already decided: a reject with more than one
  positive symbol, and anything scoring below zero (unless `allow_ham`).
  What it sends is the subject, the sender, the link domains and the
  text, cut to about `max_tokens`.
- It runs as a postfilter, within rspamd's `task_timeout` (8 s by
  default). Keep `timeout` under that and `RELAY_TIMEOUT` under `timeout`.
- Its symbols: `GPT_SPAM` up to +3 and `GPT_HAM` down to −2 (scaled by the
  probability), plus `GPT_PHISHING`, `GPT_SCAM` and `GPT_MALWARE` at +3
  each. Together they can move a message by 6 points or more; review the
  weights against your actions before letting it decide anything.
- `autolearn = true` would train bayes on the model's verdicts, and its
  mistakes would then reinforce themselves. Leave it off.
- It needs redis: in 3.14.3, without a redis server for rspamd the
  module skips every request (the log says `lua_cache.lua: missing
  required parameters for cache_get` at info level, nothing else).
  Verdicts are cached there.

**Privacy**: every message the module checks is sent to Anthropic. Make
sure that is acceptable for the mail you handle.

## Test

```sh
python -m pip install --require-hashes -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
```

The tests use a fake Anthropic client and send no requests. To try a
running relay by hand:

```sh
curl -s http://127.0.0.1:11338/v1/chat/completions \
  -H "Authorization: Bearer $RELAY_TOKEN" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"system","content":"Rate spam probability 0-1, then one line of reason."},
                   {"role":"user","content":"Subject: You won!\nClaim your prize at http://example.test"}]}'
```

## License

Copyright (C) 2026 Nic Boet

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the Free
Software Foundation, either version 3 of the License, or (at your option)
any later version. See [LICENSE](LICENSE).
