"""
Modal app: FastAPI on `lead-enrichment` exposing the integrated, chat, and
neighbour endpoints.

Deploy with `modal deploy services/app.py`. The Next.js app calls these
endpoints over HTTPS via the Modal-issued URL.

Secrets (set via `modal secret create lead-enrichment ...`):
- ANTHROPIC_API_KEY
- VOYAGE_API_KEY
- DATABASE_URL                       (Neon pooled connection string)
- UPSTASH_REDIS_REST_URL
- UPSTASH_REDIS_REST_TOKEN

The FastAPI app itself lives in `services.web` so it can also be served
directly with uvicorn for local development.
"""

from __future__ import annotations

import modal

app = modal.App("lead-enrichment")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "anthropic>=0.40.0",
        "fastapi>=0.115.0",
        "pydantic>=2.9.0",
        "psycopg[binary,pool]>=3.2.0",
        "pgvector>=0.3.0",
        "voyageai>=0.3.0",
        "lingua-language-detector>=2.0.0",
        "upstash-redis>=1.2.0",
        "sentry-sdk[fastapi]>=2.0.0",
    )
    .add_local_python_source("services")
    .add_local_dir(
        "data/exemplar_snapshots",
        remote_path="/root/data/exemplar_snapshots",
    )
)

secrets = [modal.Secret.from_name("lead-enrichment")]


@app.function(image=image, secrets=secrets, timeout=120)
@modal.asgi_app()
def fastapi_app():
    from services.web import build_app

    return build_app()
