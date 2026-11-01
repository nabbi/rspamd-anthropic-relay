# Upstream backends: direct by default, the SDK on request

Decision recorded 2026-11-01. The relay makes one kind of upstream call,
a non-streaming `POST /v1/messages`, and reads five fields from the
answer. It used to make that call through the official Anthropic SDK; it
now makes it with the Python standard library by default, and keeps the
SDK as a switch (`RELAY_BACKEND`).

| | `direct` (default) | `sdk` |
| --- | --- | --- |
| Dependencies | none | `anthropic` and its stack (pydantic, httpx2, jiter, …; `requirements-sdk.txt`) |
| Memory, idle (max RSS after import and client setup) | ~24 MiB | ~81 MiB |
| Import and client setup | ~60 ms | ~1 s |
| Throughput (4 clients, 90 KB requests, fake API on loopback) | same | same |
| `HTTPS_PROXY` and the SDK's other environment settings | no | yes |
| Code we maintain | `DirectClient` in `upstream.py` (~110 lines) | none beyond the error mapping |

Measured on Python 3.14 with anthropic 2.3.0. Of the direct backend's
24 MiB, ~8 MiB is the TLS context with the system CA store and ~8 MiB is
the interpreter. With `sdk`, ~65 MiB comes from importing the package,
mostly `anthropic.types`: pydantic models for the whole API, of which
the relay uses one call.

## Why

The systemd unit allows the relay 200 MiB (`MemoryMax`). With the SDK,
an idle relay used about 40 % of that before handling a request. The
call itself is small and stable: a JSON body, three headers, a JSON
answer. A standard-library client for it costs less to keep in step
than a dependency tree costs to install, pin and audit.

## Keeping the direct client current

We don't vendor the SDK or add it as a git submodule. Most of the SDK is
generated code, and the relay builds nothing from it. A copy of its
source would pin a version without telling us when the API moved on.
What has to stay current is narrow:

- the `anthropic-version` header (`ANTHROPIC_VERSION` in `upstream.py`)
- how the API key is sent
- the request path and body, and the response fields the relay reads

`tests/test_upstream.py` (`SdkParityTest`) runs the SDK and
`DirectClient` against the same fake API. It fails when they send a
different version, credentials header, path or body, when the SDK
starts sending a new non-telemetry header, or when the two read an
answer or an error differently. Three things run it:

- **Every push** (`sdk` job): against the SDK pinned in
  `requirements-sdk.txt`.
- **Every Monday** (`sdk-latest` job, also runnable by hand from the
  Actions tab): against the newest SDK release, unpinned. A failure here
  means the SDK changed, not that a commit broke something.
- **Dependabot**: weekly pull requests that bump the pins, each of which
  runs the parity tests.

When it fails: read the SDK's changelog for the release, and the API's
notes for the new `anthropic-version`. Then update `upstream.py`
(the constant, a header, or how the answer is read), then the pins with:

```sh
pip-compile --allow-unsafe --extra=sdk --generate-hashes \
  --output-file=requirements-sdk.txt --strip-extras pyproject.toml
```

How often it drifts: the SDK releases often (about 70 releases between
April and October 2026). The request contract changed once in that time,
at 2.0.0 (2026-10-21). `anthropic-version` went from `2023-06-01`, which
it had been since 2023, to `2026-10-21`, and the key moved from
`x-api-key` to `Authorization: Bearer`. The parity test caught the
second change the first time it ran. Expect a break about once a major
version, not once a release.

## Switching back to the SDK

Install the extra and set the backend; nothing else changes (same
settings, same log line, same answers to rspamd):

```sh
/opt/rspamd-anthropic-relay/venv/bin/pip install --require-hashes -r requirements-sdk.txt
/opt/rspamd-anthropic-relay/venv/bin/pip install --no-deps .
echo RELAY_BACKEND=sdk >> /etc/rspamd-anthropic-relay/env
```

Then raise the service's memory limit: `MemoryMax=200M` in
`contrib/systemd/rspamd-anthropic-relay.service` (the shipped value,
128M, is sized for `direct`), and restart. The startup log line names
the backend (`listening on …, backend sdk, …`). Without the SDK
installed, `RELAY_BACKEND=sdk` stops at startup with the `pip install`
to run.

Reasons to switch:

- An HTTPS proxy (`HTTPS_PROXY`), which only the SDK honours.
- A suspected difference from the SDK that the parity tests don't
  cover: `sdk` is the reference to compare against.
- The `sdk-latest` job failed and a fix to `upstream.py` isn't in yet.
  The pinned SDK sends the request the API expects for its pinned
  version.

Back to direct: `RELAY_BACKEND=direct` (or remove the line), restart, and
the SDK can be uninstalled.

## Other memory and I/O limits, same change

- **Idle connections** (`RELAY_IDLE_TIMEOUT`, 60 s): a connection that
  sends nothing, or is kept alive between requests, is closed after this
  instead of holding its thread forever.
- **Connections** (`RELAY_MAX_CONNECTIONS`, 32): one thread each. Past
  the limit a new connection gets a 503 and is closed. The unit's
  `TasksMax=64` counts threads.
- **Checks before the body**: the path, the token and a free
  `RELAY_MAX_CONCURRENT` slot are checked on the headers alone. A
  refused request costs no body read, decompression or JSON parse, and
  is answered with `Connection: close`. The parsed request is dropped
  before the upstream call. Peak memory is therefore about
  `RELAY_MAX_CONCURRENT` × (body + decompressed body + parsed request).
- **`RELAY_MAX_BODY`**: 512 KiB instead of 1 MiB. 100,000 characters
  (`RELAY_MAX_INPUT_CHARS`) of UTF-8 text fit with room to spare.
- **One write per response**: headers and body are buffered and sent
  together, with `TCP_NODELAY`, so a kept-alive connection doesn't wait
  on Nagle's algorithm and the client's delayed ACK.
- **Upstream pool**: at most `RELAY_MAX_CONCURRENT` idle connections to
  the API with either backend (the SDK's default allows 100). The direct
  client reuses one for up to 30 s. It checks that the server has not
  closed it before reusing it, and retries once on a fresh connection
  when a reused one turns out to be closed.
