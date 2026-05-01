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

All FastAPI / Pydantic code lives inside `fastapi_app()` so it runs only inside
the container image, not on the deploy host.
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
    )
    .add_local_python_source("services")
)

secrets = [modal.Secret.from_name("lead-enrichment")]


@app.function(image=image, secrets=secrets, timeout=120)
@modal.asgi_app()
def fastapi_app():
    import json
    from typing import Any

    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel

    from services.chat import chat_stream
    from services.embeddings import find_neighbours
    from services.enrich import enrich_lead
    from services.ratelimit import check_and_consume
    from services.validation import ValidationError

    class EnrichRequest(BaseModel):
        profile: str
        company: str | None = None
        example_id: str | None = None

    class ChatMessage(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: list[ChatMessage]
        example_id: str | None = None

    class NeighboursRequest(BaseModel):
        text: str
        k: int = 3

    web = FastAPI(title="lead-enrichment")

    def _client_ip(request: Request) -> str:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def _gate(request: Request, bucket: str) -> None:
        allowed, _ = await check_and_consume(_client_ip(request), bucket=bucket)
        if not allowed:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    @web.post("/enrich")
    async def enrich_endpoint(payload: EnrichRequest, request: Request) -> dict[str, Any]:
        await _gate(request, bucket="enrich")
        try:
            return await enrich_lead(
                payload.profile, payload.company, example_id=payload.example_id
            )
        except ValidationError as e:
            raise HTTPException(
                status_code=400, detail={"code": e.code, "message": e.message}
            )

    @web.post("/chat")
    async def chat_endpoint(payload: ChatRequest, request: Request) -> StreamingResponse:
        await _gate(request, bucket="chat")
        messages = [m.model_dump() for m in payload.messages]

        async def event_source():
            try:
                async for event in chat_stream(messages, example_id=payload.example_id):
                    yield f"data: {json.dumps(event)}\n\n"
            except ValidationError as e:
                yield (
                    "data: "
                    + json.dumps({"type": "error", "code": e.code, "message": e.message})
                    + "\n\n"
                )

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @web.post("/neighbours")
    async def neighbours_endpoint(payload: NeighboursRequest, request: Request) -> dict:
        await _gate(request, bucket="neighbours")
        return {"neighbours": await find_neighbours(payload.text, k=payload.k)}

    @web.get("/healthz")
    async def healthz() -> dict:
        return {"ok": True}

    return web
