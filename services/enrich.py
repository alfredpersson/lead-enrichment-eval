"""
Integrated build: enrich_lead.

One Anthropic call with extended thinking on, the integrated system prompt
(cached), and the `enrich_lead` tool with strict schema. Returns the structured
output plus eval-neighbour context and telemetry-grade meta.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any

from anthropic import AsyncAnthropic

from services.embeddings import embed, find_neighbours
from services.prompts import ENRICH_LEAD_TOOL, integrated_system_blocks
from services.telemetry import write_request_row
from services.validation import check_input

MODEL_ID = "claude-sonnet-4-6"
THINKING_BUDGET_TOKENS = 4000
MAX_OUTPUT_TOKENS = 4096

_client: AsyncAnthropic | None = None


def _get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        _client = AsyncAnthropic(api_key=api_key)
    return _client


def _user_message(profile: str, company: str | None) -> str:
    if company:
        return (
            f"<profile>\n{profile}\n</profile>\n\n"
            f"<company>\n{company}\n</company>"
        )
    return f"<profile>\n{profile}\n</profile>"


def _grounded_count(claims: list[dict], input_text: str) -> int:
    """Whitespace-normalised, case-insensitive substring match for source quotes."""
    if not claims:
        return 0
    norm_input = " ".join(input_text.split()).lower()
    count = 0
    for claim in claims:
        quote = " ".join((claim.get("source_quote") or "").split()).lower()
        if quote and quote in norm_input:
            count += 1
    return count


async def enrich_lead(
    profile: str,
    company: str | None = None,
    *,
    example_id: str | None = None,
) -> dict[str, Any]:
    """
    Run the integrated build end-to-end and return a serialisable dict matching
    the EnrichOutput shape from the plan, with `meta` populated.
    """
    input_lang = check_input(profile, company)
    request_id = str(uuid.uuid4())
    input_text = profile if not company else f"{profile}\n\n{company}"
    started = time.perf_counter()

    client = _get_client()
    response = await client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_OUTPUT_TOKENS,
        thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET_TOKENS},
        system=integrated_system_blocks(),
        tools=[ENRICH_LEAD_TOOL],
        tool_choice={"type": "tool", "name": "enrich_lead"},
        messages=[{"role": "user", "content": _user_message(profile, company)}],
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    tool_block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_block is None:
        raise RuntimeError("Model did not call the enrich_lead tool")
    output: dict = dict(tool_block.input)

    embedding = await embed(input_text)
    neighbours = await find_neighbours(input_text, k=3)

    usage = response.usage
    cache_hit = bool(getattr(usage, "cache_read_input_tokens", 0))
    thinking_tokens = getattr(usage, "thinking_tokens", None)

    claims = output.get("claims", [])
    grounded = _grounded_count(claims, input_text)

    await write_request_row(
        {
            "request_id": request_id,
            "mode": "integrated",
            "example_id": example_id,
            "input_lang": input_lang,
            "input_char_count": len(input_text),
            "model_id": MODEL_ID,
            "thinking_enabled": True,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "thinking_tokens": thinking_tokens,
            "cache_hit": cache_hit,
            "latency_ms": latency_ms,
            "embedding": embedding,
            "action": output.get("action"),
            "fit_score": output.get("fit_score", {}).get("value"),
            "claim_count": len(claims),
            "claims_with_source_quote_count": grounded,
        }
    )

    output["meta"] = {
        "request_id": request_id,
        "latency_ms": latency_ms,
        "tokens_in": usage.input_tokens,
        "tokens_out": usage.output_tokens,
        "thinking_tokens": thinking_tokens,
        "cache_hit": cache_hit,
        "model": MODEL_ID,
        "eval_neighbours": neighbours,
    }
    return output
