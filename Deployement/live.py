"""Live telemetry connector architecture (production boundary).

Design rules (non-negotiable):
  - Connectors are ADAPTERS: this module defines the interfaces,
    timeouts, retries, and failure semantics. It does NOT invent any
    vendor API (no fake EDR/SIEM/firewall calls anywhere).
  - Secrets are NEVER hardcoded and NEVER stored on config objects.
    Connectors take the NAME of an environment variable holding the
    credential and resolve it once at construction; a missing variable
    is an explicit, actionable error naming the variable (never its
    value).
  - Failures are LOUD: ``CollectorError`` on connection failure,
    malformed payloads, timeouts, and partial-host translation errors
    (recorded per-host, never silently dropped). A failing collector
    must never look like a quiet network -- see pipeline health
    handling and item 6 (freshness semantics).
  - Only the standard library is used (urllib), so no new dependencies.

Provided implementations:
  - ``HttpJsonCollector``: polls one HTTPS/HTTP JSON endpoint with a
    caller-supplied ``translate`` function (the vendor-specific
    boundary). Pass ``insecure_skip_verify=True`` ONLY for lab hosts;
    it is refused unless explicitly enabled and is always logged.
  - ``ResilientCollector``: wraps any collector with bounded retries +
    backoff on transient CollectorError. Retries NEVER apply to
    destructive enforcement -- this wrapper is telemetry-only.
  - ``CollectorHealth``: ok/degraded/down bookkeeping owned by the
    pipeline (consecutive-failure counters, last error, timestamps).
"""

import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field

from .telemetry import HostTelemetry, SecurityEvent, TelemetryBatch, TelemetryCollector  # noqa: E501


HEALTH_DOWN_AFTER = 3  # consecutive failures before health reads "down"

# Default cap for one HTTP telemetry payload. A full 112-host JSON
# document is kilobytes; 8 MiB leaves wide headroom while bounding a
# malicious/broken endpoint's memory impact. Configurable per site.
DEFAULT_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects instead of following them.

    Rationale: this collector attaches a bearer token to every request.
    urllib's default redirect handling would resend that Authorization
    header to wherever a (possibly attacker-controlled) Location points,
    including http:// downgrades. A legitimate endpoint move is a
    one-line config change (``collector.endpoint``); silent credential
    forwarding is never acceptable.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CollectorError(
            f"refusing redirect ({code}) to {newurl!r}: update "
            f"collector.endpoint to the canonical URL instead")


class CollectorError(RuntimeError):
    """Telemetry could not be obtained or parsed. Never quiet."""


class SecretMissingError(CollectorError):
    """A required credential environment variable is not set."""


@dataclass
class EndpointConfig:
    """Non-secret connector settings. Secrets live ONLY in env vars."""
    name: str = "live"
    endpoint: str = ""          # e.g. "https://siem.local/api/hosts"
    timeout_s: float = 10.0
    poll_interval_s: float = 5.0  # minimum gap between live polls; honoured
    # by HttpJsonCollector (one-shot/file/mock collectors never sleep)
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    token_env: str = ""         # NAME of env var holding the bearer token
    insecure_skip_verify: bool = False  # lab only; refused unless explicit


def resolve_secret(env_name, purpose):
    """Read a credential from the environment (value never logged).

    Raises SecretMissingError naming the VARIABLE (not the value) when
    unset or blank. Callers must never print/return the resolved value.
    """
    if not env_name or not isinstance(env_name, str):
        raise SecretMissingError(
            f"{purpose}: no credential env var configured "
            f"(set 'token_env' to a variable NAME, not a secret)")
    value = os.environ.get(env_name, "")
    if not value:
        raise SecretMissingError(
            f"{purpose}: environment variable {env_name!r} is not set. "
            f"Export it before starting the collector; it is never "
            f"written to config files or logs.")
    return value


def scrub_secrets(obj):
    """Redact credential-looking values from structures bound for logs.

    Any mapping key containing token/secret/password/api_key/cookie/
    authorization (case-insensitive) has its value replaced. Lists,
    tuples, and nested mappings are handled recursively. Strings are
    returned unchanged (callers must not interpolate secrets into
    free text in the first place).
    """
    needles = ("token", "secret", "password", "api_key", "apikey",
               "cookie", "authorization")
    if isinstance(obj, dict):
        return {k: ("***REDACTED***"
                    if any(n in str(k).lower() for n in needles)
                    else scrub_secrets(v))
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        cleaned = [scrub_secrets(v) for v in obj]
        return type(obj)(cleaned) if isinstance(obj, tuple) else cleaned
    return obj


class HostTranslateError(CollectorError):
    """One host's payload could not be translated (others continue)."""


@dataclass
class CollectorHealth:
    state: str = "ok"  # ok | degraded | down
    consecutive_failures: int = 0
    last_error: str = ""
    last_success_ts: float = 0.0
    total_batches: int = 0
    total_failures: int = 0

    def record_success(self, now):
        self.consecutive_failures = 0
        self.last_error = ""
        self.last_success_ts = float(now)
        self.total_batches += 1
        self.state = "ok"

    def record_failure(self, error):
        self.consecutive_failures += 1
        self.total_failures += 1
        self.last_error = str(error)[:500]
        self.state = ("down" if self.consecutive_failures >= HEALTH_DOWN_AFTER
                      else "degraded")

    def as_dict(self):
        return {"state": self.state,
                "consecutive_failures": self.consecutive_failures,
                "last_error": self.last_error,
                "last_success_ts": self.last_success_ts,
                "total_batches": self.total_batches,
                "total_failures": self.total_failures}


class HttpJsonCollector(TelemetryCollector):
    """Poll a JSON endpoint; vendor parsing injected via ``translate``.

    ``translate(payload)`` is THE vendor boundary: it receives the
    decoded JSON document and must return a ``dict`` mapping host key
    -> ``HostTelemetry`` (it may raise HostTranslateError per host, or
    CollectorError for a document-level problem). This class owns only
    transport: GET, bearer auth from env, timeout, JSON decode.
    """

    def __init__(self, config, translate, clock=None, sleeper=None):
        """``clock``/``sleeper`` are injectable (default monotonic/sleep)
        so poll pacing is deterministic and testable without real waits.
        """
        import time as _time
        if not isinstance(config, EndpointConfig):
            raise CollectorError(
                f"HttpJsonCollector needs EndpointConfig, got {type(config)}")
        if not callable(translate):
            raise CollectorError("HttpJsonCollector needs a translate() "
                                 "callable (the vendor adapter boundary)")
        if not config.endpoint:
            raise CollectorError("HttpJsonCollector needs a non-empty "
                                 "endpoint URL")
        if config.timeout_s is None or float(config.timeout_s) <= 0:
            raise CollectorError("timeout_s must be a positive number")
        if config.poll_interval_s is None or float(config.poll_interval_s) < 0:  # noqa: E501
            raise CollectorError("poll_interval_s must be >= 0")
        if (config.max_response_bytes is None
                or int(config.max_response_bytes) <= 0):
            raise CollectorError("max_response_bytes must be a positive "
                                 "byte count")
        self.config = config
        self._translate = translate
        self._clock = clock or _time.monotonic
        self._sleeper = sleeper or _time.sleep
        self._last_poll = None
        self._token = (resolve_secret(config.token_env, config.name)
                       if config.token_env else None)
        if config.insecure_skip_verify:
            import warnings
            warnings.warn(f"{config.name}: TLS verification disabled "
                          f"(lab use only)")

    @property
    def exhausted(self):
        return False  # live streams never end; failures raise instead

    def _request(self):
        import ssl
        req = urllib.request.Request(self.config.endpoint, method="GET")
        req.add_header("Accept", "application/json")
        if self._token is not None:
            # Set on the per-request object only; the token is never
            # stored on shared state, config files, or log records.
            req.add_header("Authorization", f"Bearer {self._token}")
        context = None
        if self.config.insecure_skip_verify:
            context = ssl._create_unverified_context()
        # Custom opener (NOT the global urlopen): redirects are refused
        # outright (see _NoRedirectHandler) so bearer credentials can
        # never follow a Location header off-host or down to http://.
        handlers = [_NoRedirectHandler(),
                    urllib.request.HTTPHandler(),
                    urllib.request.HTTPSHandler(context=context)]
        opener = urllib.request.build_opener(*handlers)
        try:
            with opener.open(req,
                             timeout=self.config.timeout_s) as resp:
                # Bounded read: at most max+1 bytes, so an oversized
                # body is rejected BEFORE unbounded memory use.
                raw = resp.read(int(self.config.max_response_bytes) + 1)
                if len(raw) > int(self.config.max_response_bytes):
                    raise CollectorError(
                        f"{self.config.name}: response exceeded "
                        f"max_response_bytes="
                        f"{int(self.config.max_response_bytes)} "
                        f"({len(raw)}+ bytes); refusing to parse")
        except urllib.error.HTTPError as exc:
            raise CollectorError(
                f"{self.config.name}: HTTP {exc.code} from "
                f"{self.config.endpoint}") from exc
        except urllib.error.URLError as exc:
            raise CollectorError(
                f"{self.config.name}: connection failed for "
                f"{self.config.endpoint}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise CollectorError(
                f"{self.config.name}: timed out after "
                f"{self.config.timeout_s}s") from exc
        except Exception as exc:
            if isinstance(exc, CollectorError):
                raise
            raise CollectorError(
                f"{self.config.name}: transport failure: {exc}") from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise CollectorError(
                f"{self.config.name}: malformed JSON payload "
                f"({len(raw)} bytes): {exc}") from exc

    def _pace_poll(self):
        """Sleep only the remainder of poll_interval_s (never negative).

        First poll is immediate; a slow request already covering the
        interval sleeps nothing. One-shot/file/mock collectors never
        call this -- pacing applies to repeated live polling only.
        """
        interval = float(self.config.poll_interval_s)
        if interval <= 0:
            return
        now = self._clock()
        if self._last_poll is not None:
            remaining = self._last_poll + interval - now
            if remaining > 0:
                self._sleeper(remaining)
                now = self._clock()
        self._last_poll = now

    def next_batch(self):
        self._pace_poll()
        payload = self._request()
        try:
            hosts = self._translate(payload)
        except CollectorError:
            raise
        except Exception as exc:
            raise CollectorError(
                f"{self.config.name}: translator failed: {exc}") from exc
        if not isinstance(hosts, dict):
            raise CollectorError(
                f"{self.config.name}: translator must return "
                f"dict[key, HostTelemetry], got {type(hosts)}")
        clean, partial_errors = {}, []
        for key, item in hosts.items():
            if isinstance(item, HostTelemetry):
                clean[key] = item
            elif isinstance(item, HostTranslateError):
                partial_errors.append({"key": key, "error": str(item)})
            else:
                partial_errors.append(
                    {"key": key,
                     "error": f"translator returned {type(item)}"})
        return TelemetryBatch(timestamp=time.time(), hosts=clean,
                              source=self.config.name,
                              partial_errors=partial_errors)

    def close(self):
        self._token = None


class ResilientCollector(TelemetryCollector):
    """Bounded retries + backoff around another collector (telemetry only).

    Retries apply to transient CollectorError from POLLING. This wrapper
    must never be used to retry destructive enforcement -- it only
    re-asks for observations.
    """

    def __init__(self, inner, retries=2, backoff_s=1.0):
        self.inner = inner
        self.retries = int(retries)
        self.backoff_s = float(backoff_s)
        if self.retries < 0 or self.backoff_s < 0:
            raise CollectorError("retries/backoff_s must be >= 0")

    @property
    def exhausted(self):
        return bool(getattr(self.inner, "exhausted", False))

    def next_batch(self):
        attempts = 1 + self.retries
        last = None
        for attempt in range(attempts):
            try:
                return self.inner.next_batch()
            except CollectorError as exc:
                last = exc
                if attempt + 1 < attempts and self.backoff_s > 0:
                    time.sleep(self.backoff_s * (attempt + 1))
        raise last

    def close(self):
        try:
            self.inner.close()
        except Exception:
            pass
