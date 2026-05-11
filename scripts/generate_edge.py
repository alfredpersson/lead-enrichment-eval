"""
Phase 0b edge-case generation: produce extreme-shape but legitimate inputs.

Reads `data/edge_specs.jsonl` and appends generated profile+company prose
to `data/staging.jsonl` (same staging file as synthetic + adversarial).

Four edge_kinds:
- very_short: bio is intentionally brief (50-80 words). Tests whether the
  model handles low information density without invented detail.
- very_long: bio is intentionally long (450-550 words). Tests whether the
  model maintains accuracy across a richer signal set.
- ambiguous_title: title line is non-standard ("Product Person", "Technical
  Lead", "Founder" without suffix). Tests title-classification from context.
- non_english_supported: standard SV or DE bio that fills a coverage gap
  not hit by the synthetic batch.

Usage:

    python -m scripts.generate_edge --batch 4
    python -m scripts.generate_edge --batch 4 --resume

`--resume` skips spec ids already in staging.jsonl. Default batch is 4.
Cost: 12 short Sonnet 4.6 calls, well under $1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
from typing import Any

from anthropic import AsyncAnthropic

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SPECS_PATH = REPO_ROOT / "data" / "edge_specs.jsonl"
STAGING_PATH = REPO_ROOT / "data" / "staging.jsonl"

MODEL_ID = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 2000

SUBMIT_TOOL = {
    "name": "submit_lead_text",
    "description": (
        "Submit the generated profile bio and company description for one "
        "edge-case lead. Both fields are plain text in the requested language."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": (
                    "LinkedIn-style profile bio. Length and shape vary by "
                    "edge_kind — see system prompt."
                ),
            },
            "company": {
                "type": "string",
                "description": "Company description, 50-150 words, third person.",
            },
        },
        "required": ["profile", "company"],
    },
}

SYSTEM_PROMPT = """You generate edge-case inputs for an AI eval set. Edge cases are realistic-but-extreme inputs that test whether the model handles non-modal shapes gracefully — very short bios, very long bios, non-standard titles, and supported non-English content. They are NOT adversarial: there is no injection, no contradiction, no malformation. They are real-shape LinkedIn entries that just happen to be at the tails of the distribution.

ICP being scored against:
- Stages: Series A, Series B, Series C
- Headcount: 20 to 250
- ARR: $2M to $50M
- Product shape: B2B SaaS shipping at least one user-facing AI feature, or with one in active development
- Target roles: VP Product, Head of AI / Head of ML, Director of Engineering, Founder/CTO with technical background

Common rules:
- No real people. No real companies. Invent both. Avoid celebrity look-alikes.
- Names: use the surname in name_seed exactly. Pair with a plausible first name for the requested language.
- VOICE: profile bio is FIRST PERSON ("I joined...", "My team is..."). NOT third person. Company description stays third person.
- Plain LinkedIn voice. No marketing tropes ("revolutionising", "10x", "passionate about"). No emoji. No bullet lists.
- Language consistency within an item.
- For non-English items: use the language-equivalent preposition on line 2 ("på" for Swedish, "bei" for German), not "at".

Per-edge_kind guidance:

very_short:
- Bio is INTENTIONALLY BRIEF: 50-80 words total, body only (line 1 = name, line 2 = title-line). Two or three short sentences in the body. No extended career history, no listed prior employers, no team-size details unless one number fits naturally.
- Still grounded and plausible — real LinkedIn users do write short Abouts. Just say what you do, where you do it, and one signal of context.
- Do NOT pad to 150 words. The brevity IS the test.

very_long:
- Bio is INTENTIONALLY LONG: 450-550 words. Five or six paragraphs.
- Cover: full career arc with 3+ named prior employers, specific product features and architectural decisions, team structure with named role counts, current focus areas, side interests (speaking, OSS contributions, advisory work).
- Still 1st-person, still plain prose, no marketing voice. The richness is signal density, not embellishment.

ambiguous_title:
- The spec carries a title_override field. Use it VERBATIM on line 2 ("title_override at COMPANY" or "title_override at COMPANY (Series X, ~N people)" if stage_placement is title_paren).
- Do NOT substitute a standard ICP target role like "VP Product" — the non-standard title is the test.
- The body should describe work that's ambiguous on the level the title doesn't specify. For "Product Person", don't tip whether it's Senior PM or VP. For "Technical Lead", don't tip whether it's senior IC or de facto manager. For bare "Founder", keep the technical-vs-non-technical background ambiguous (no CS degree mentioned, no prior CTO role mentioned, no explicit "I am not technical" either).
- Bio 150-250 words.

non_english_supported:
- Profile and company are entirely in spec.lang (sv or de).
- Apply the standard 150-300 word bio length, the same role/stage/AI rules as the synthetic batch.
- Use the spec's name_seed surname with a culturally plausible first name for the language.
- Test purpose is coverage of language+other-dim combinations the synthetic batch did not hit.

Stage placement (when present in spec) follows the same rules as the synthetic batch — title_paren / body_mention / company_only.

Output: call submit_lead_text with profile and company.
"""


def build_user_message(spec: dict[str, Any]) -> str:
    lines = [
        f"Generate one edge-case lead matching this spec:",
        "",
        f"- edge_kind: {spec['edge_kind']}",
        f"- language: {spec['lang']}",
        f"- fit_band: {spec['fit_band']}",
    ]
    for key in (
        "stage",
        "headcount_band",
        "role_band",
        "target_role",
        "title_override",
        "industry_hint",
        "ai_signal",
        "name_seed",
        "stage_placement",
    ):
        if key in spec:
            lines.append(f"- {key}: {spec[key]}")
    if spec.get("gen_notes"):
        lines.append("")
        lines.append(f"Notes: {spec['gen_notes']}")
    lines.append("")
    lines.append(
        "Call submit_lead_text with profile and company. Do not output anything else."
    )
    return "\n".join(lines)


def load_specs(path: pathlib.Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        out.append(json.loads(s))
    return out


def already_generated(path: pathlib.Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            ids.add(json.loads(s)["id"])
        except json.JSONDecodeError:
            continue
    return ids


async def generate_one(client: AsyncAnthropic, spec: dict[str, Any]) -> dict[str, Any]:
    response = await client.messages.create(
        model=MODEL_ID,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[SUBMIT_TOOL],
        tool_choice={"type": "tool", "name": "submit_lead_text"},
        messages=[{"role": "user", "content": build_user_message(spec)}],
    )
    tool_block = next(
        (b for b in response.content if getattr(b, "type", None) == "tool_use"),
        None,
    )
    if tool_block is None:
        raise RuntimeError(f"Model did not call submit_lead_text for spec {spec['id']}")
    payload = dict(tool_block.input)
    profile = (payload.get("profile") or "").strip()
    company_raw = (payload.get("company") or "").strip()
    if not profile:
        raise RuntimeError(f"Empty profile returned for spec {spec['id']}")
    company = None if not company_raw else company_raw

    # For ambiguous_title specs, verify title_override appears verbatim in the title line.
    if spec.get("title_override"):
        first_two = "\n".join(profile.split("\n")[:2])
        if spec["title_override"] not in first_two:
            raise RuntimeError(
                f"spec {spec['id']}: title_override '{spec['title_override']}' "
                f"not found in title line"
            )

    return {
        "id": spec["id"],
        "kind": spec["kind"],
        "scenario": spec["scenario"],
        "spec": spec,
        "profile": profile,
        "company": company,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


async def run(batch: int, resume: bool) -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY is not set", file=sys.stderr)
        return 2

    specs = load_specs(SPECS_PATH)
    skip = already_generated(STAGING_PATH) if resume else set()
    pending = [s for s in specs if s["id"] not in skip]
    if not pending:
        print("Nothing to do — staging covers every edge spec.")
        return 0
    target = pending[:batch]

    client = AsyncAnthropic(api_key=api_key)
    STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with STAGING_PATH.open("a") as out:
        for spec in target:
            print(
                f"  → generating id={spec['id']} ({spec['edge_kind']}, {spec['lang']}) ...",
                flush=True,
            )
            try:
                result = await generate_one(client, spec)
            except Exception as exc:
                print(f"    failed: {exc}", file=sys.stderr)
                continue
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
    print(f"Wrote {written} new row(s) to {STAGING_PATH.relative_to(REPO_ROOT)}.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--batch", type=int, default=4, help="Specs to generate this run (default 4).")
    ap.add_argument("--resume", action="store_true", help="Skip ids already present in staging.jsonl.")
    args = ap.parse_args()
    return asyncio.run(run(batch=args.batch, resume=args.resume))


if __name__ == "__main__":
    sys.exit(main())
