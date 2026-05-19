"""
FastAPI app construction, factored out so it can be served either inside the
Modal container (`services/app.py`) or directly with uvicorn for local dev
(`uvicorn services.web:app --reload --port 8000`).
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services.chat import chat_stream
from services.embeddings import find_neighbours
from services.enrich import enrich_lead_stream
from services.ratelimit import check_and_consume
from services.validation import ValidationError


def _init_sentry() -> None:
    """Initialise Sentry from SENTRY_DSN env var. No-op when unset."""
    dsn = os.environ.get("SENTRY_DSN")
    if not dsn:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=dsn,
        traces_sample_rate=0.1,
        environment=os.environ.get("APP_ENV", "local"),
        send_default_pii=False,
    )


_init_sentry()


class EnrichRequest(BaseModel):
    profile: str
    company: str | None = None
    example_id: str | None = None
    bypass_cache: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatContext(BaseModel):
    lead_name: str | None = None
    profile: str | None = None
    company: str | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    example_id: str | None = None
    context: ChatContext | None = None
    bypass_cache: bool = False


class NeighboursRequest(BaseModel):
    text: str
    k: int = 3


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def build_app() -> FastAPI:
    web = FastAPI(title="lead-enrichment")

    async def _gate(request: Request, bucket: str) -> None:
        allowed, _ = await check_and_consume(_client_ip(request), bucket=bucket)
        if not allowed:
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

    @web.post("/enrich")
    async def enrich_endpoint(payload: EnrichRequest, request: Request) -> StreamingResponse:
        await _gate(request, bucket="enrich")

        async def event_source():
            try:
                async for event in enrich_lead_stream(
                    payload.profile,
                    payload.company,
                    example_id=payload.example_id,
                    bypass_cache=payload.bypass_cache,
                ):
                    yield f"data: {json.dumps(event)}\n\n"
            except ValidationError as e:
                yield (
                    "data: "
                    + json.dumps({"type": "error", "code": e.code, "message": e.message})
                    + "\n\n"
                )

        return StreamingResponse(event_source(), media_type="text/event-stream")

    @web.post("/chat")
    async def chat_endpoint(payload: ChatRequest, request: Request) -> StreamingResponse:
        await _gate(request, bucket="chat")
        messages = [m.model_dump() for m in payload.messages]

        context = payload.context.model_dump() if payload.context else None

        async def event_source():
            try:
                async for event in chat_stream(
                    messages,
                    example_id=payload.example_id,
                    context=context,
                    bypass_cache=payload.bypass_cache,
                ):
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


app = build_app()
