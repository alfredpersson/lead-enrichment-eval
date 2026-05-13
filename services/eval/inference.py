"""
Anthropic inference drivers for the eval runner.

Default path is live (non-batch) calls with a bounded asyncio semaphore so we
capture true per-call latency for the scorecard's p50/p95 metric. The plan
calls for the batch API for cost, but at the v1 scope (60-80 items × 2 modes
≈ 146 calls + perturbation pass) live cost is small and per-call latency is
load-bearing for the scorecard. A `submit_batch` helper is kept here as a
toggle for the larger test-set sizes the eval will accept post-v1.

Both modes hold the same Sonnet 4.6 model constant during eval: thinking on
for integrated, off for chat. This isolates the model-side contract from the
underlying model itself, matching what production runs see.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from anthropic import AsyncAnthropic

from services.eval.dataset import EvalItem
from services.prompts import (
    CHAT_SYSTEM_PROMPT,
    ENRICH_LEAD_TOOL,
    INTEGRATED_SYSTEM_PROMPT,
)

INTEGRATED_MODEL_ID = "claude-sonnet-4-6"
CHAT_MODEL_ID = "claude-sonnet-4-6"
INTEGRATED_THINKING_BUDGET = 4000
INTEGRATED_MAX_TOKENS = 4096
CHAT_MAX_TOKENS = 2048

LIVE_CONCURRENCY = 2
REQUEST_TIMEOUT_S = 120
MAX_RETRIES = 6


@dataclass
class InferenceResult:
    item_id: str
    success: bool
    output: dict[str, Any] | None
    text: str | None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    thinking_tokens: int | None
    cache_read_tokens: int
    error: str | None
    raw_stop_reason: str | None


@dataclass
class ChatResult:
    """Result of a multi-turn chat eval for one item.

    `text` is the concatenation of all assistant turns. `turns_used` counts
    user messages sent (1..max_turns). `cap_hit` is True when the loop hit
    the cap without the extractor flagging extractor_complete; cap-hits are
    scored as failures per the plan.
    """

    item_id: str
    success: bool
    text: str | None
    transcript: list[dict[str, str]]
    latency_ms: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    turns_used: int
    cap_hit: bool
    error: str | None


def _client() -> AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return AsyncAnthropic(api_key=api_key, timeout=REQUEST_TIMEOUT_S, max_retries=MAX_RETRIES)


def _user_message(profile: str, company: str | None) -> str:
    if company:
        return (
            f"<profile>\n{profile}\n</profile>\n\n"
            f"<company>\n{company}\n</company>"
        )
    return f"<profile>\n{profile}\n</profile>"


def _integrated_system() -> list[dict]:
    return [
        {
            "type": "text",
            "text": INTEGRATED_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _chat_system() -> list[dict]:
    return [
        {
            "type": "text",
            "text": CHAT_SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _parse_integrated(item_id: str, response: Any, latency_ms: int) -> InferenceResult:
    tool_block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    output = dict(tool_block.input) if tool_block is not None else None
    usage = response.usage
    return InferenceResult(
        item_id=item_id,
        success=output is not None,
        output=output,
        text=None,
        latency_ms=latency_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        thinking_tokens=getattr(usage, "thinking_tokens", None),
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        error=None if output is not None else "no tool_use block",
        raw_stop_reason=getattr(response, "stop_reason", None),
    )


def _parse_chat(item_id: str, response: Any, latency_ms: int) -> InferenceResult:
    text_parts = [
        getattr(b, "text", "") or ""
        for b in response.content
        if getattr(b, "type", None) == "text"
    ]
    text = "".join(text_parts)
    usage = response.usage
    return InferenceResult(
        item_id=item_id,
        success=bool(text),
        output=None,
        text=text,
        latency_ms=latency_ms,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        thinking_tokens=None,
        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
        error=None if text else "empty response",
        raw_stop_reason=getattr(response, "stop_reason", None),
    )


async def _call_integrated(
    client: AsyncAnthropic,
    item: EvalItem,
    sem: asyncio.Semaphore,
) -> InferenceResult:
    async with sem:
        started = time.perf_counter()
        try:
            response = await client.messages.create(
                model=INTEGRATED_MODEL_ID,
                max_tokens=INTEGRATED_MAX_TOKENS,
                thinking={
                    "type": "enabled",
                    "budget_tokens": INTEGRATED_THINKING_BUDGET,
                },
                system=_integrated_system(),
                tools=[ENRICH_LEAD_TOOL],
                tool_choice={"type": "auto"},
                messages=[
                    {
                        "role": "user",
                        "content": _user_message(item.profile, item.company),
                    }
                ],
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            return InferenceResult(
                item_id=item.id,
                success=False,
                output=None,
                text=None,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=None,
                cache_read_tokens=0,
                error=f"{type(e).__name__}: {e}",
                raw_stop_reason=None,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return _parse_integrated(item.id, response, latency_ms)


async def _call_chat(
    client: AsyncAnthropic,
    item: EvalItem,
    sem: asyncio.Semaphore,
) -> InferenceResult:
    async with sem:
        started = time.perf_counter()
        try:
            response = await client.messages.create(
                model=CHAT_MODEL_ID,
                max_tokens=CHAT_MAX_TOKENS,
                system=_chat_system(),
                messages=[
                    {
                        "role": "user",
                        "content": _user_message(item.profile, item.company),
                    }
                ],
            )
        except Exception as e:
            latency_ms = int((time.perf_counter() - started) * 1000)
            return InferenceResult(
                item_id=item.id,
                success=False,
                output=None,
                text=None,
                latency_ms=latency_ms,
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=None,
                cache_read_tokens=0,
                error=f"{type(e).__name__}: {e}",
                raw_stop_reason=None,
            )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return _parse_chat(item.id, response, latency_ms)


async def run_integrated(
    items: list[EvalItem],
    *,
    client: AsyncAnthropic | None = None,
    on_done: "Callable[[InferenceResult], None] | None" = None,
) -> list[InferenceResult]:
    """Run the integrated build across `items` in parallel.

    When `client` is provided it is reused and not closed (the caller owns
    its lifecycle). When None a client is built and closed locally.

    `on_done` (when provided) is called with each result as it completes —
    used by the runner for progress logging.
    """
    if not items:
        return []
    owned = client is None
    client = client or _client()
    sem = asyncio.Semaphore(LIVE_CONCURRENCY)

    async def _wrap(item: EvalItem) -> InferenceResult:
        r = await _call_integrated(client, item, sem)
        if on_done is not None:
            on_done(r)
        return r

    try:
        return await asyncio.gather(*[_wrap(item) for item in items])
    finally:
        if owned:
            await client.close()


async def run_chat(
    items: list[EvalItem],
    *,
    client: AsyncAnthropic | None = None,
    on_done: "Callable[[InferenceResult], None] | None" = None,
) -> list[InferenceResult]:
    """Same lifecycle contract as `run_integrated`. Single-turn — used by
    the legacy code path and tests. For the metric required by the plan
    (`steps-to-completion`, capped at 3) see `run_chat_multiturn`."""
    if not items:
        return []
    owned = client is None
    client = client or _client()
    sem = asyncio.Semaphore(LIVE_CONCURRENCY)

    async def _wrap(item: EvalItem) -> InferenceResult:
        r = await _call_chat(client, item, sem)
        if on_done is not None:
            on_done(r)
        return r

    try:
        return await asyncio.gather(*[_wrap(item) for item in items])
    finally:
        if owned:
            await client.close()


CHAT_FOLLOWUP_PROMPT = (
    "Thanks. Please make sure your answer covers, for this lead: "
    "industry, segment, seniority, company size, an ICP fit score on the "
    "0.0–1.0 scale with the five named dimensions, a list of claims each "
    "with a verbatim source quote from the input, a draft outreach hook, "
    "and a recommended action (auto_add, propose, discard, or refuse). "
    "Fill in any of those that were missing."
)


async def _multiturn_one(
    client: AsyncAnthropic,
    item: EvalItem,
    sem: asyncio.Semaphore,
    max_turns: int,
    extract_and_check: "Callable[[EvalItem, str], Awaitable[bool]]",
) -> ChatResult:
    """Run the chat across up to `max_turns` user turns for one item.

    `extract_and_check(item, latest_chat_text)` must return True when the
    extractor reports `extractor_complete` for the current accumulated
    response. The loop stops on the first True or on the cap.
    """
    transcript: list[dict[str, str]] = [
        {"role": "user", "content": _user_message(item.profile, item.company)}
    ]
    aggregated_text_parts: list[str] = []
    total_in = total_out = cache_read = 0
    started = time.perf_counter()
    cap_hit = False
    turns_used = 0
    last_error: str | None = None
    last_stop_reason: str | None = None

    for turn in range(1, max_turns + 1):
        turns_used = turn
        async with sem:
            try:
                response = await client.messages.create(
                    model=CHAT_MODEL_ID,
                    max_tokens=CHAT_MAX_TOKENS,
                    system=_chat_system(),
                    messages=transcript,
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                cap_hit = True
                break
        text_parts = [
            getattr(b, "text", "") or ""
            for b in response.content
            if getattr(b, "type", None) == "text"
        ]
        turn_text = "".join(text_parts)
        aggregated_text_parts.append(turn_text)
        transcript.append({"role": "assistant", "content": turn_text})
        usage = response.usage
        total_in += usage.input_tokens
        total_out += usage.output_tokens
        cache_read += getattr(usage, "cache_read_input_tokens", 0) or 0
        last_stop_reason = getattr(response, "stop_reason", None)
        if not turn_text:
            last_error = "empty assistant turn"
            cap_hit = True
            break

        aggregated = "\n\n".join(aggregated_text_parts)
        complete = await extract_and_check(item, aggregated)
        if complete:
            break
        if turn == max_turns:
            cap_hit = True
            break
        transcript.append({"role": "user", "content": CHAT_FOLLOWUP_PROMPT})

    latency_ms = int((time.perf_counter() - started) * 1000)
    success = not cap_hit and last_error is None
    return ChatResult(
        item_id=item.id,
        success=success,
        text="\n\n".join(aggregated_text_parts) if aggregated_text_parts else None,
        transcript=transcript,
        latency_ms=latency_ms,
        input_tokens=total_in,
        output_tokens=total_out,
        cache_read_tokens=cache_read,
        turns_used=turns_used,
        cap_hit=cap_hit,
        error=last_error,
    )


async def run_chat_multiturn(
    items: list[EvalItem],
    *,
    extract_and_check: "Callable[[EvalItem, str], Awaitable[bool]]",
    max_turns: int = 3,
    client: AsyncAnthropic | None = None,
    on_done: "Callable[[ChatResult], None] | None" = None,
) -> list[ChatResult]:
    """Run multi-turn chat with extractor-driven stopping.

    For each item, ask the chat model the lead question, then call
    `extract_and_check` on the assistant's reply. If the extractor reports
    incomplete coverage, send a generic follow-up prompt asking for the
    missing fields. Continue up to `max_turns` user turns; cap-hit is
    scored as failure per the plan's steps-to-completion metric.
    """
    if not items:
        return []
    owned = client is None
    client = client or _client()
    sem = asyncio.Semaphore(LIVE_CONCURRENCY)

    async def _wrap(item: EvalItem) -> ChatResult:
        r = await _multiturn_one(client, item, sem, max_turns, extract_and_check)
        if on_done is not None:
            on_done(r)
        return r

    try:
        return await asyncio.gather(*[_wrap(item) for item in items])
    finally:
        if owned:
            await client.close()
