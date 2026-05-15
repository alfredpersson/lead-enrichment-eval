"""
Chat build: free-form streaming chat against Sonnet 4.6.

No tools, no structured-output schema, no claim-grounding rule. The system
prompt describes the task in prose. Telemetry is written on stream completion;
extractor_complete is left null here and set by the eval harness when it runs
the structured extractor pass.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic

from services.config import is_local
from services.embeddings import embed, find_neighbours
from services.prompts import chat_system_blocks
from services.telemetry import write_request_row
from services.validation import check_input

MODEL_ID = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 2048

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = AsyncAnthropic(api_key=api_key)
    return _client


async def chat_stream(
    messages: list[dict[str, Any]],
    *,
    example_id: str | None = None,
    profile_for_validation: str | None = None,
    company_for_validation: str | None = None,
    context: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """
    Stream tokens from Sonnet for the chat build. Yields events:
      {"type": "text", "delta": "..."}
      {"type": "done", "meta": {...}}

    When `context` is provided, the lead's profile/company is injected as a
    second system block so the assistant has the active record without the
    user needing to paste it.
    """
    if context is not None:
        input_lang = check_input(
            context.get("profile") or "",
            context.get("company") or None,
        )
    elif profile_for_validation is not None:
        input_lang = check_input(profile_for_validation, company_for_validation)
    else:
        first_user = next(
            (m for m in messages if m.get("role") == "user"),
            None,
        )
        text = (first_user or {}).get("content", "") if first_user else ""
        input_lang = check_input(text if isinstance(text, str) else "", None)

    request_id = str(uuid.uuid4())
    turn_count = sum(1 for m in messages if m.get("role") == "user")
    input_chars = sum(
        len(m.get("content", "")) if isinstance(m.get("content"), str) else 0
        for m in messages
    )
    started = time.perf_counter()
    output_text_parts: list[str] = []
    usage_in = usage_out = 0
    cache_hit = False

    system_blocks = chat_system_blocks()
    if context:
        lead_name = (context.get("lead_name") or "").strip()
        profile = (context.get("profile") or "").strip()
        company = (context.get("company") or "").strip()
        lines = ["Active lead the user is asking about:"]
        if lead_name:
            lines.append(f"Name: {lead_name}")
        if profile:
            lines.append(f"Profile:\n{profile}")
        if company:
            lines.append(f"Company:\n{company}")
        lines.append(
            "Answer with this lead in mind. If the user asks comparative "
            "questions referencing other leads from earlier in the "
            "conversation, draw on what was discussed."
        )
        system_blocks = [
            *system_blocks,
            {"type": "text", "text": "\n\n".join(lines)},
        ]

    client = _get_client()
    async with client.messages.stream(
        model=MODEL_ID,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system_blocks,
        messages=messages,
    ) as stream:
        async for chunk in stream.text_stream:
            output_text_parts.append(chunk)
            yield {"type": "text", "delta": chunk}
        final = await stream.get_final_message()
        usage_in = final.usage.input_tokens
        usage_out = final.usage.output_tokens
        cache_hit = bool(getattr(final.usage, "cache_read_input_tokens", 0))

    latency_ms = int((time.perf_counter() - started) * 1000)
    output_text = "".join(output_text_parts)

    if is_local():
        embedding = None
    else:
        embed_text = output_text or " ".join(
            m.get("content", "") for m in messages if isinstance(m.get("content"), str)
        )
        embedding = await embed(embed_text or " ")

    # Anchor neighbours to the lead profile (not the assistant reply) so the
    # panel matches the integrated build's "test-set items similar to this lead"
    # framing and stays comparable across modes.
    if context:
        lead_parts = [
            (context.get("profile") or "").strip(),
            (context.get("company") or "").strip(),
        ]
        neighbour_text = "\n\n".join(p for p in lead_parts if p)
    else:
        neighbour_text = ""
    neighbours = await find_neighbours(neighbour_text or output_text or " ", k=3)

    await write_request_row(
        {
            "request_id": request_id,
            "mode": "chat",
            "example_id": example_id,
            "input_lang": input_lang,
            "input_char_count": input_chars,
            "model_id": MODEL_ID,
            "thinking_enabled": False,
            "input_tokens": usage_in,
            "output_tokens": usage_out,
            "cache_hit": cache_hit,
            "latency_ms": latency_ms,
            "embedding": embedding,
            "turn_count": turn_count,
            "extractor_complete": None,
        }
    )

    yield {
        "type": "done",
        "meta": {
            "request_id": request_id,
            "latency_ms": latency_ms,
            "tokens_in": usage_in,
            "tokens_out": usage_out,
            "cache_hit": cache_hit,
            "model": MODEL_ID,
            "turn_count": turn_count,
            "eval_neighbours": neighbours,
        },
    }
