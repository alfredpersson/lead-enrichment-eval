"""
Phase 0b synthesis: generate prose for each spec in `data/synthesis_specs.jsonl`.

Reads spec rows, calls Anthropic with a tool-use schema that returns just
profile + company text (no gold labels), and appends each result to
`data/staging.jsonl`. The labeller hand-fills gold values from the staging file
and moves rows into `data/exemplars.json` with the next sequential id.

Usage:

    python -m scripts.generate_synthetic --batch 6
    python -m scripts.generate_synthetic --batch 6 --resume

`--resume` skips spec ids already present in staging.jsonl. Default batch is 6
items (matches the agreed cadence; pause for review between batches).

Cost note: ~36 short Sonnet 4.6 calls. At list price (~$3/M input, $15/M output)
the full run costs well under $1.
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
SPECS_PATH = REPO_ROOT / "data" / "synthesis_specs.jsonl"
STAGING_PATH = REPO_ROOT / "data" / "staging.jsonl"

MODEL_ID = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 1500

SUBMIT_TOOL = {
    "name": "submit_lead_text",
    "description": (
        "Submit the generated profile bio and company description for one "
        "synthetic lead. Both fields are plain text in the requested language."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "profile": {
                "type": "string",
                "description": (
                    "LinkedIn-style profile bio. First line is the person's name, "
                    "second line is their title and company, then a body of "
                    "150-300 words written in first person. Plausible and "
                    "specific; no generic marketing voice."
                ),
            },
            "company": {
                "type": "string",
                "description": (
                    "Company description, 50-150 words, third person. "
                    "Empty string if the spec sets company_optional=true."
                ),
            },
        },
        "required": ["profile", "company"],
    },
}

SYSTEM_PROMPT = """You generate synthetic LinkedIn-style profile bios and short B2B company descriptions for an AI eval set. Output is consumed by a hand-labelling step, so realism matters more than variety alone.

ICP being scored against (this is fixed across the entire eval set):
- Stages: Series A, Series B, Series C
- Headcount: 20 to 250
- ARR: $2M to $50M
- Product shape: B2B SaaS company shipping at least one user-facing AI feature, or with one in active development
- Target roles (verbatim, in priority order):
    * VP Product
    * Head of AI / Head of ML
    * Director of Engineering
    * Founder/CTO with a technical background

role_band values and what they mean for the title line of the bio:
- target: the title line MUST name one of the four ICP target roles. If the spec carries a target_role field, use that exact role title (e.g., "VP Product", "Head of AI", "Director of Engineering"). Do not substitute "VP People Analytics", "Head of Growth", "VP Customer Success", "VP Engineering" or any other VP-level title — those are NOT target roles even if they sound senior.
- adjacent: a role one tier off — Senior PM with AI focus, Staff Engineer at small co, VP Engineering, OR Founder/CEO at a target-shape company without confirmed technical background. The gen_notes will specify which adjacency.
- founder_nontech: Founder/CEO at a target-shape company without confirmed technical background. Bio must NOT establish a technical background (no engineering/CS history, no prior CTO role, no degree in CS).
- ic_tech: a technical IC at target seniority without decision authority (e.g., "Senior Software Engineer", "Staff Engineer at a large co").
- mismatch: the spec's industry_hint defines the role — e.g., "VP Operations at a consultancy", "Marketing Manager at a B2C app", "Freelance designer". Do not pick an ICP target role here.

fit_band values and what they require:
- strong: every dimension lands cleanly inside the ICP — stage in target, headcount in range, ARR plausibly in range, explicit shipped AI feature, role on the ICP target list. Auto-add territory.
- ambiguous: one or two dimensions partial — see the spec's gen_notes for which. Propose territory.
- weak: clear miss on product shape, role, or stage. Discard territory. Bio should still be plausible.
- refuse: bio is intentionally so sparse that no ICP dimension can be judged. One or two lines, no industry, no stage, no role detail, no AI signal. Looks like a real but uninformative LinkedIn headline.

Rules:
- No real people. No real companies. Invent both. Avoid celebrity look-alikes.
- Names: use the surname provided in name_seed exactly. Pair it with a plausible first name for the requested language. The name should not feel anglicised in non-English specs.
- Profile bio is first-person, 150-300 words for strong/ambiguous/weak fits and intentionally short (10-40 words) for refuse cases. First line: full name. Second line: "TITLE at COMPANY" (or the language equivalent: "på" for Swedish, "bei" for German). Body: a believable LinkedIn About-section in plain prose. Mention concrete details (team size, recent product moves, prior employers if natural, geography). No bullet lists. No marketing tropes ("revolutionising", "10x", "passionate about", "I help X do Y"). No emoji.
- Funding stage and headcount: the company description is where these live as primary fact. The bio's placement is determined by the spec's stage_placement field:
    * "title_paren" — show stage and/or headcount in a terse parenthetical right after the company name on line 2: "VP Product at Lattice Forge (Series B, ~120 people)" or "(Series B)" or "(~120 people)" — vary which.
    * "body_mention" — line 2 is just "TITLE at COMPANY" with no parens; instead, mention the stage briefly somewhere in the body as an offhand clause: "we closed our Series A last year", "Series B from Atomico", "I joined just before the Series B".
    * "company_only" — the bio does not mention stage or headcount at all. Those signals live only in the company text. The bio describes the role, the work, and the person without naming the funding round or the headcount.
  If stage_placement is missing, default to no parens and no body mention (company text carries it). Do NOT write company-About-page sentences in the bio like "X is a Series B HR tech company focused on...". The bio is a person's voice, not a press release.
- Company description is third-person, 50-150 words. Funding stage, headcount, geography, customer segment, what the product does. No marketing voice. If company_optional is true, return an empty string for company.
- Match the spec exactly. fit_band, role_band, target_role (when present), and ai_signal are non-negotiable. Do not soften weak fits or strengthen ambiguous ones. Do not pick a role that drifts off-spec.
- Language is the language of BOTH profile and company. Do not mix languages within an item. The output will be classified and scored in English regardless, so do not translate.
- Adversarial cases (prompt injection / jailbreak / contradictory signals) are NOT in this batch. If a spec asks for one, refuse with an empty profile.
- Stay specific and grounded. A "Series B B2B SaaS HR tech with attrition prediction" should name a plausible product surface ("we forecast attrition risk by manager team", "our model integrates with Workday and BambooHR"), not "we leverage AI to transform HR".
"""


def build_user_message(spec: dict[str, Any]) -> str:
    """Render a spec as a directive for the generator."""
    lines = [
        f"Generate one synthetic lead matching this spec:",
        "",
        f"- language: {spec['lang']}",
        f"- fit_band: {spec['fit_band']}",
    ]
    for key in (
        "stage",
        "headcount_band",
        "role_band",
        "target_role",
        "industry_hint",
        "ai_signal",
        "name_seed",
        "stage_placement",
    ):
        if key in spec:
            lines.append(f"- {key}: {spec[key]}")
    if spec.get("company_optional"):
        lines.append("- company_optional: true (return empty string for company)")
    if spec.get("gen_notes"):
        lines.append("")
        lines.append(f"Notes: {spec['gen_notes']}")
    lines.append("")
    lines.append(
        "Call submit_lead_text with the profile and company. Do not output "
        "anything else."
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
        ids.add(json.loads(s)["id"])
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
    company = None if (spec.get("company_optional") and not company_raw) else (company_raw or None)
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
        print("Nothing to do — staging covers every spec.")
        return 0
    target = pending[:batch]

    client = AsyncAnthropic(api_key=api_key)
    STAGING_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with STAGING_PATH.open("a") as out:
        for spec in target:
            print(f"  → generating id={spec['id']} ({spec['fit_band']}, {spec['lang']}) ...", flush=True)
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
    ap.add_argument("--batch", type=int, default=6, help="Number of specs to generate this run (default 6).")
    ap.add_argument("--resume", action="store_true", help="Skip ids already present in staging.jsonl.")
    args = ap.parse_args()
    return asyncio.run(run(batch=args.batch, resume=args.resume))


if __name__ == "__main__":
    sys.exit(main())
