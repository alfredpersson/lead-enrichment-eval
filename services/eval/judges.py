"""
LLM-as-judge passes: claim grounding and hook coherence.

Claim grounding runs two judges (Claude Sonnet 4.6 and an OpenAI flagship) per
the methodology's cross-provider plan, then publishes both rates plus Cohen's
kappa for inter-judge agreement. Only claims that pass the deterministic
whitespace/case-insensitive substring check enter the judge pool; substring
failures are scored ungrounded automatically.

Hook coherence runs a single OpenAI judge as a binary pass/fail with a
written critique — no Likert scale. Pass criteria are baked into the prompt
so re-runs are stable. The critique is the load-bearing artifact: a pass
without specific evidence in the critique is not a real pass.

OpenAI calls are optional: when OPENAI_API_KEY is unset the OpenAI judge
returns `None` for each claim and Cohen's kappa is omitted. The Anthropic
grounding rate still publishes. The methodology page makes this explicit so
prospects don't mistake a missing judge for a missing metric.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from anthropic import AsyncAnthropic

from services.eval.dataset import EvalItem
from services.eval.metrics import _normalise

ANTHROPIC_GROUNDING_JUDGE_MODEL = "claude-sonnet-4-6"
OPENAI_GROUNDING_JUDGE_MODEL = os.environ.get("OPENAI_GROUNDING_MODEL", "gpt-5")
OPENAI_HOOK_JUDGE_MODEL = os.environ.get("OPENAI_HOOK_MODEL", "gpt-5-mini")

JUDGE_CONCURRENCY = 4
JUDGE_TIMEOUT_S = 60
JUDGE_MAX_RETRIES = 6


GROUNDING_SYSTEM = """\
You judge whether a claim about a B2B sales lead is supported by the lead's
input. You will see the input text and one claim with a source quote that
the model picked from the input.

Reply with a single JSON object: {"grounded": true|false, "reason": "..."}.

A claim is grounded only if the source quote actually appears in the input
*and* the claim's content is consistent with what that quote says in context.
Paraphrase is allowed; new facts not present in the input are not.
"""

HOOK_SYSTEM = """\
You judge whether a salesperson's draft outreach hook passes or fails for B2B
outreach. Binary outcome with a written critique — no scales.

A hook PASSES when ALL of these hold:
- it includes multiple specifics that come verbatim or paraphrased from the
  lead's input (specifics invented by the model do not count);
- the tone is professional and appropriate for B2B outreach;
- it is on-topic and coherent.

A hook FAILS when ANY of these hold:
- it is incoherent, off-topic, or generic with no input-grounded specifics;
- it has only a single specific, or specifics that are not actually in the
  input;
- the tone is over-familiar, salesy, dismissive, or otherwise inappropriate;
- the action is `discard` or `refuse` (no hook should be drafted for these).

Reply with a single JSON object: {"passes": true|false, "critique": "..."}.
The critique is required for both outcomes — name the specific evidence
(quote or paraphrase the parts that pass or fail). A terse "looks fine" is
not a valid critique.
"""


@dataclass
class ClaimJudgement:
    item_id: str
    claim_index: int
    claim_text: str
    source_quote: str
    substring_match: bool
    opus_grounded: bool | None
    openai_grounded: bool | None
    opus_reason: str | None = None
    openai_reason: str | None = None


@dataclass
class HookJudgement:
    item_id: str
    passes: bool | None
    critique: str | None


@dataclass
class GroundingResults:
    judgements: list[ClaimJudgement] = field(default_factory=list)
    opus_rate: float | None = None
    openai_rate: float | None = None
    headline_rate: float | None = None
    kappa: float | None = None
    n_claims: int = 0
    n_judged: int = 0


@dataclass
class HookResults:
    judgements: list[HookJudgement] = field(default_factory=list)
    pass_rate: float | None = None
    n_scored: int = 0


def _anthropic() -> AsyncAnthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return AsyncAnthropic(api_key=api_key, timeout=JUDGE_TIMEOUT_S, max_retries=JUDGE_MAX_RETRIES)


def _openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import AsyncOpenAI
    except ImportError:
        return None
    return AsyncOpenAI(api_key=api_key, timeout=JUDGE_TIMEOUT_S)


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _grounding_user_prompt(item: EvalItem, claim: dict[str, Any]) -> str:
    return (
        f"<input>\n{item.input_text}\n</input>\n\n"
        f"<claim>{claim.get('text', '')}</claim>\n"
        f"<source_quote>{claim.get('source_quote', '')}</source_quote>"
    )


async def _opus_judge_claim(
    client: AsyncAnthropic,
    item: EvalItem,
    claim: dict[str, Any],
    sem: asyncio.Semaphore,
) -> tuple[bool | None, str | None]:
    async with sem:
        try:
            resp = await client.messages.create(
                model=ANTHROPIC_GROUNDING_JUDGE_MODEL,
                max_tokens=300,
                system=GROUNDING_SYSTEM,
                messages=[{"role": "user", "content": _grounding_user_prompt(item, claim)}],
            )
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    text = "".join(
        getattr(b, "text", "") or ""
        for b in resp.content
        if getattr(b, "type", None) == "text"
    )
    payload = _parse_json_payload(text)
    if not payload:
        return None, "judge returned non-JSON"
    return bool(payload.get("grounded")), str(payload.get("reason", ""))


async def _openai_judge_claim(
    client: Any,
    model_id: str,
    item: EvalItem,
    claim: dict[str, Any],
    sem: asyncio.Semaphore,
) -> tuple[bool | None, str | None]:
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": GROUNDING_SYSTEM},
                    {"role": "user", "content": _grounding_user_prompt(item, claim)},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            return None, f"{type(e).__name__}: {e}"
    text = resp.choices[0].message.content if resp.choices else ""
    payload = _parse_json_payload(text or "")
    if not payload:
        return None, "judge returned non-JSON"
    return bool(payload.get("grounded")), str(payload.get("reason", ""))


def _cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Two-rater binary kappa."""
    if len(a) != len(b) or not a:
        return None
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_true = sum(1 for x in a if x) / n
    pb_true = sum(1 for x in b if x) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


async def judge_grounding(
    items_by_id: dict[str, EvalItem],
    outputs_by_id: dict[str, dict[str, Any]],
    *,
    anthropic_client: AsyncAnthropic | None = None,
    openai_client: Any | None = None,
) -> GroundingResults:
    """Run both judges (when keys available) over every claim in every output.

    Substring failures auto-fail (no judge call). Only substring-passing
    claims hit the judges, so cost scales with claim count, not item count.
    """
    flat: list[tuple[str, int, dict[str, Any]]] = []
    for item_id, output in outputs_by_id.items():
        if not output:
            continue
        for idx, claim in enumerate(output.get("claims") or []):
            flat.append((item_id, idx, claim))

    if not flat:
        return GroundingResults()

    owned_anthropic = anthropic_client is None
    anthropic_client = anthropic_client or _anthropic()
    owned_openai = openai_client is None
    if openai_client is None:
        openai_client = _openai_client()
    openai_model = OPENAI_GROUNDING_JUDGE_MODEL
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

    judgements: list[ClaimJudgement] = [
        ClaimJudgement(
            item_id=item_id,
            claim_index=idx,
            claim_text=claim.get("text", ""),
            source_quote=claim.get("source_quote", ""),
            substring_match=_normalise(claim.get("source_quote") or "")
            in _normalise(items_by_id[item_id].input_text),
            opus_grounded=None,
            openai_grounded=None,
        )
        for item_id, idx, claim in flat
    ]
    claim_lookup = {(it, ix): cl for (it, ix, cl) in flat}

    async def opus_for(j: ClaimJudgement) -> tuple[bool | None, str | None]:
        if not j.substring_match:
            return False, "substring mismatch"
        return await _opus_judge_claim(
            anthropic_client,
            items_by_id[j.item_id],
            claim_lookup[(j.item_id, j.claim_index)],
            sem,
        )

    async def openai_for(j: ClaimJudgement) -> tuple[bool | None, str | None]:
        # Conditional-client check goes first: when no OpenAI key is configured
        # the judge cannot be consulted at all, including on substring-fail
        # claims. The previous order populated openai_grounded=False for
        # substring failures even with no key, which pulled openai_rate to 0
        # and (via min over rates) tanked headline_rate.
        if openai_client is None:
            return None, None
        if not j.substring_match:
            return False, "substring mismatch"
        return await _openai_judge_claim(
            openai_client,
            openai_model,
            items_by_id[j.item_id],
            claim_lookup[(j.item_id, j.claim_index)],
            sem,
        )

    try:
        opus_outcomes = await asyncio.gather(*[opus_for(j) for j in judgements])
        openai_outcomes = await asyncio.gather(*[openai_for(j) for j in judgements])
    finally:
        if owned_anthropic:
            await anthropic_client.close()
        if owned_openai and openai_client is not None and hasattr(openai_client, "close"):
            close = getattr(openai_client, "close")
            if asyncio.iscoroutinefunction(close):
                await close()

    for j, (opus_g, opus_r), (oa_g, oa_r) in zip(
        judgements, opus_outcomes, openai_outcomes
    ):
        j.opus_grounded = opus_g
        j.opus_reason = opus_r
        j.openai_grounded = oa_g
        j.openai_reason = oa_r

    opus_judged = [j for j in judgements if j.opus_grounded is not None]
    openai_judged = [j for j in judgements if j.openai_grounded is not None]
    opus_rate = (
        sum(1 for j in opus_judged if j.opus_grounded) / len(opus_judged)
        if opus_judged
        else None
    )
    openai_rate = (
        sum(1 for j in openai_judged if j.openai_grounded) / len(openai_judged)
        if openai_judged
        else None
    )
    # Kappa is only meaningful on claims that both judges actually evaluated.
    # Substring failures auto-set both judges to False ("substring mismatch")
    # — including them inflates kappa with trivial agreements.
    paired = [
        j for j in judgements
        if j.substring_match
        and j.opus_grounded is not None
        and j.openai_grounded is not None
    ]
    kappa = _cohen_kappa(
        [bool(j.opus_grounded) for j in paired],
        [bool(j.openai_grounded) for j in paired],
    )
    rates = [r for r in (opus_rate, openai_rate) if r is not None]
    headline_rate = min(rates) if rates else None
    return GroundingResults(
        judgements=judgements,
        opus_rate=opus_rate,
        openai_rate=openai_rate,
        headline_rate=headline_rate,
        kappa=kappa,
        n_claims=len(judgements),
        n_judged=len(opus_judged),
    )


def _hook_user_prompt(item: EvalItem, hook_text: str, action: str | None) -> str:
    return (
        f"<input>\n{item.input_text}\n</input>\n\n"
        f"<action>{action or 'unknown'}</action>\n"
        f"<hook>{hook_text}</hook>"
    )


def _parse_hook_payload(item_id: str, payload: dict[str, Any] | None) -> HookJudgement:
    if not payload:
        return HookJudgement(
            item_id=item_id, passes=None, critique="judge returned non-JSON"
        )
    passes = payload.get("passes")
    critique = str(payload.get("critique", "")).strip()
    if isinstance(passes, bool):
        return HookJudgement(item_id=item_id, passes=passes, critique=critique)
    return HookJudgement(
        item_id=item_id,
        passes=None,
        critique="passes field missing or not boolean",
    )


async def _hook_judge_one(
    client: Any,
    model_id: str,
    item: EvalItem,
    output: dict[str, Any],
    sem: asyncio.Semaphore,
) -> HookJudgement:
    draft_hook_obj = (output or {}).get("draft_hook")
    if isinstance(draft_hook_obj, dict):
        hook_text = draft_hook_obj.get("text") or ""
    elif isinstance(draft_hook_obj, str):
        hook_text = draft_hook_obj
    else:
        hook_text = ""
    action = (output or {}).get("action")
    if not hook_text:
        return HookJudgement(item_id=item.id, passes=None, critique="empty hook")
    async with sem:
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": HOOK_SYSTEM},
                    {"role": "user", "content": _hook_user_prompt(item, hook_text, action)},
                ],
                response_format={"type": "json_object"},
            )
        except Exception as e:
            return HookJudgement(
                item_id=item.id, passes=None, critique=f"{type(e).__name__}: {e}"
            )
    text = resp.choices[0].message.content if resp.choices else ""
    payload = _parse_json_payload(text or "")
    return _parse_hook_payload(item.id, payload)


async def judge_hooks(
    items_by_id: dict[str, EvalItem],
    outputs_by_id: dict[str, dict[str, Any] | None],
    *,
    openai_client: Any | None = None,
) -> HookResults:
    """Judge every output's draft_hook as pass/fail with a written critique.

    Requires OPENAI_API_KEY. When the key is absent we return empty results
    so the runner can still write a snapshot without hook numbers. Caller
    may inject a shared `AsyncOpenAI` instance via `openai_client`.
    """
    owned = openai_client is None
    if openai_client is None:
        openai_client = _openai_client()
    if openai_client is None:
        return HookResults()
    model_id = OPENAI_HOOK_JUDGE_MODEL
    sem = asyncio.Semaphore(JUDGE_CONCURRENCY)
    tasks = []
    for item_id, output in outputs_by_id.items():
        if output is None:
            continue
        item = items_by_id[item_id]
        tasks.append(_hook_judge_one(openai_client, model_id, item, output, sem))
    judgements = await asyncio.gather(*tasks) if tasks else []
    if owned and hasattr(openai_client, "close"):
        close = getattr(openai_client, "close")
        if asyncio.iscoroutinefunction(close):
            await close()
    decided = [j.passes for j in judgements if j.passes is not None]
    pass_rate = sum(1 for p in decided if p) / len(decided) if decided else None
    return HookResults(
        judgements=judgements, pass_rate=pass_rate, n_scored=len(decided)
    )
