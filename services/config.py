"""
Runtime mode flag.

`APP_ENV=local` (default) skips external infra that isn't strictly needed to
exercise the Anthropic-facing surfaces — Upstash rate limiting, Postgres
telemetry writes, and pgvector neighbour lookups all no-op. Direct Anthropic
and Voyage calls still happen so the integrated and chat flows behave end-to-
end against the live model.

`APP_ENV=prod` (set on Modal via the `lead-enrichment` secret) requires every
external dependency to be configured.
"""

from __future__ import annotations

import os


def _resolve() -> str:
    return os.environ.get("APP_ENV", "local").strip().lower()


def is_local() -> bool:
    return _resolve() == "local"


def is_prod() -> bool:
    return _resolve() == "prod"
