"""
Chat extractor pass.

The chat build's free-form output is parsed back into the same structured
schema the integrated build emits, so both modes can score against the same
gold labels. Per the plan this runs on Haiku 4.5 with a strict tool-call
extractor schema. Failures (missing fields, refusal to use the tool, etc) are
preserved on the result so the scorecard can surface extraction completeness
as its own metric.

The extractor sees the *chat output*, not the original input. It cannot
hallucinate grounding; if the chat output didn't include a claim with a
source quote, no extracted claim has one either. That's the load-bearing
asymmetry the cross-mode comparison surfaces.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from anthropic import AsyncAnthropic

from services.eval.dataset import EvalItem
from services.eval.inference import InferenceResult
from services.prompts import ENRICH_LEAD_TOOL

EXTRACTOR_MODEL_ID = "claude-haiku-4-5-20251001"
EXTRACTOR_MAX_TOKENS = 2048
EXTRACTOR_CONCURRENCY = 6
EXTRACTOR_TIMEOUT_S = 60
EXTRACTOR_MAX_RETRIES = 6

EXTRACTOR_SYSTEM = """\
You read a salesperson's free-form notes about a B2B lead and extract the
structured assessment the salesperson described. Call the `enrich_lead`
tool when the notes contain an assessment to extract.

Rules:
- Only include claims the notes actually make. Do not invent.
- A claim's source_quote must be a verbatim substring of the *lead's input*
  shown in the <input> block, not the notes. If the notes do not point to a
  quote in the input, set source_quote to "".
- If the notes refuse to assess (e.g. "I can't help with that", "I won't
  score this"), call the tool with action="refuse", an empty claims list,
  fit_score.value=0.0, and an empty draft_hook.text. This is the only path
  for unparseable refusals — do not skip the tool call.
- If the notes are not an assessment at all (e.g. small talk, a question
  back to the user), reply in plain text WITHOUT calling the tool. The
  caller will treat that as an extraction failure.
- If the notes don't pick an action but describe an assessment, choose the
  action the notes most clearly imply.
- If the notes don't give a fit score, infer the value the notes most
  clearly imply on the 0.0 to 1.0 scale.
"""

REFUSAL_HINTS = (
    "i can't help",
    "i cannot help",
    "i won't",
    "i will not",
    "i refuse",
    "unable to assist",
    "cannot assist",
)


@dataclass
class ExtractorResult:
    item_id: str
    success: bool
    output: dict[str, Any] | None
    extractor_complete: bool
    error: str | None
    input_tokens: int
    output_tokens: int
    # When the extractor declined to emit a tool call (chat output didn't
    # contain an assessment), the raw chat text is captured so the scorecard
    # can surface it under failure modes. None when the extractor succeeded.
    notes_unparsed: str | None = None


def _client() -> AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return AsyncAnthropic(api_key=api_key, timeout=EXTRACTOR_TIMEOUT_S, max_retries=EXTRACTOR_MAX_RETRIES)


def _user_message(item: EvalItem, chat_text: str) -> str:
    profile = item.profile
    company = item.company or ""
    parts = [f"<input>\n<profile>\n{profile}\n</profile>"]
    if company:
        parts.append(f"<company>\n{company}\n</company>")
    parts.append("</input>")
    parts.append(f"\n<notes>\n{chat_text}\n</notes>")
    return "\n".join(parts)


def _is_complete(output: dict[str, Any] | None) -> bool:
    """The plan's `extractor_complete`: did the chat output cover all gold-shape
    structured fields? Heuristic: classification + fit_score.value + at least
    one claim + an action + a draft_hook.text. Missing claims_used or hook
    text on `discard`/`refuse` actions is allowed."""
    if not output:
        return False
    cls = output.get("classification") or {}
    fs = output.get("fit_score") or {}
    claims = output.get("claims") or []
    hook = output.get("draft_hook") or {}
    action = output.get("action")
    has_cls = all(cls.get(k) for k in ("industry", "segment", "seniority", "company_size"))
    has_fs = isinstance(fs.get("value"), (int, float))
    has_action = action in {"auto_add", "propose", "discard", "refuse"}
    hook_required = action in {"auto_add", "propose"}
    has_hook = bool(hook.get("text")) if hook_required else True
    has_claims = bool(claims) if hook_required else True
    return has_cls and has_fs and has_action and has_hook and has_claims


async def extract_one_text(
    client: AsyncAnthropic,
    item: EvalItem,
    chat_text: str | None,
    sem: asyncio.Semaphore,
    *,
    fallback_error: str | None = None,
) -> ExtractorResult:
    """Run the extractor against `chat_text` (the accumulated assistant
    output) and return a parsed `ExtractorResult`. Public so the multi-turn
    chat loop can call it per turn to drive its stop condition.

    `fallback_error` is recorded when the chat text is empty/missing.
    """
    if not chat_text:
        return ExtractorResult(
            item_id=item.id,
            success=False,
            output=None,
            extractor_complete=False,
            error=fallback_error or "chat output empty",
            input_tokens=0,
            output_tokens=0,
        )
    async with sem:
        try:
            response = await client.messages.create(
                model=EXTRACTOR_MODEL_ID,
                max_tokens=EXTRACTOR_MAX_TOKENS,
                system=EXTRACTOR_SYSTEM,
                tools=[ENRICH_LEAD_TOOL],
                # `auto` lets the extractor decline when the chat text is
                # not an assessment. Forced tool_choice would have it
                # hallucinate an action even on refusals/non-sequiturs.
                tool_choice={"type": "auto"},
                messages=[
                    {"role": "user", "content": _user_message(item, chat_text)}
                ],
            )
        except Exception as e:
            return ExtractorResult(
                item_id=item.id,
                success=False,
                output=None,
                extractor_complete=False,
                error=f"{type(e).__name__}: {e}",
                input_tokens=0,
                output_tokens=0,
            )
    tool_block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    text_parts = [
        getattr(b, "text", "") or ""
        for b in response.content
        if getattr(b, "type", None) == "text"
    ]
    refusal_text = "".join(text_parts).strip()

    if tool_block is not None:
        output = dict(tool_block.input)
        return ExtractorResult(
            item_id=item.id,
            success=True,
            output=output,
            extractor_complete=_is_complete(output),
            error=None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    # No tool call. If the chat output looked like a clean refusal, emit a
    # synthesised refuse-shaped output so the action accuracy metric counts
    # the refusal correctly. Otherwise expose the unparsed notes.
    lowered = (chat_text or "").lower()
    if any(hint in lowered for hint in REFUSAL_HINTS):
        synth = {
            "classification": {
                "industry": "",
                "segment": "",
                "seniority": "IC",
                "company_size": "1-10",
            },
            "fit_score": {
                "value": 0.0,
                "dimensions": {
                    "stage_match": 0.0,
                    "headcount_match": 0.0,
                    "arr_match": 0.0,
                    "product_shape_match": 0.0,
                    "role_match": 0.0,
                },
            },
            "claims": [],
            "draft_hook": {"text": "", "claims_used": [], "confidence": 0.0},
            "action": "refuse",
            "reasoning": "chat declined to assess; extractor synthesised refuse",
        }
        return ExtractorResult(
            item_id=item.id,
            success=True,
            output=synth,
            extractor_complete=True,
            error=None,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

    return ExtractorResult(
        item_id=item.id,
        success=False,
        output=None,
        extractor_complete=False,
        error="extractor returned no tool_use (chat output not an assessment)",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        notes_unparsed=refusal_text or chat_text,
    )


def make_completeness_check(
    client: AsyncAnthropic,
    sem: asyncio.Semaphore,
) -> "Callable[[EvalItem, str], Awaitable[bool]]":
    """Build the multi-turn loop's stop-condition callback.

    Returns an async function that runs the extractor on `(item, chat_text)`
    and reports True when `extractor_complete` fires. Reuses one client and
    one semaphore so concurrent multi-turn loops share rate-limit capacity.
    """

    async def _check(item: EvalItem, chat_text: str) -> bool:
        result = await extract_one_text(client, item, chat_text, sem)
        return result.extractor_complete

    return _check


async def extract_chat_outputs(
    items: list[EvalItem],
    chat_results: list[InferenceResult],
    *,
    client: AsyncAnthropic | None = None,
    on_done: "Callable[[ExtractorResult], None] | None" = None,
) -> list[ExtractorResult]:
    """Run the extractor pass on each chat output, preserving item order.

    Same client lifecycle contract as `inference.run_integrated`: caller may
    inject a shared `AsyncAnthropic` to avoid open/close overhead.
    """
    by_id = {r.item_id: r for r in chat_results}
    owned = client is None
    client = client or _client()
    sem = asyncio.Semaphore(EXTRACTOR_CONCURRENCY)

    async def _wrap(item: EvalItem) -> ExtractorResult:
        ir = by_id[item.id]
        text = ir.text if ir.success else None
        fallback = ir.error if not ir.success else None
        r = await extract_one_text(
            client, item, text, sem, fallback_error=fallback
        )
        if on_done is not None:
            on_done(r)
        return r

    try:
        return await asyncio.gather(*[_wrap(item) for item in items])
    finally:
        if owned:
            await client.close()
