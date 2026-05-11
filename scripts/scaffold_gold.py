"""
Scaffold gold-label structure for each row in `data/staging.jsonl`.

For each synthetic item, pre-fills what's deterministic or extractable from
the bio + company text — leaving only the rubric calls (product_shape_match,
role_match), the holistic fit_score.value, the claims with verbatim source
quotes, and a few categorical labels for the labeller. Refuse cases are
fully scaffolded (all-zero dims, action='refuse', no claims).

Workflow:

    1. python -m scripts.generate_synthetic --batch 6 [--resume]   (already run)
    2. python -m scripts.scaffold_gold
       → writes data/staging_with_gold.jsonl
    3. Labeller opens staging_with_gold.jsonl in an editor, fills in every
       null field and the empty arrays, removes the _todo and _extracted
       helper fields, then runs:
    4. python -m scripts.promote_gold
       → moves fully-labelled rows into data/exemplars.json under their
       sequential ids and bumps version to v1.0 once all 36 land.

Pre-filled deterministically:
- input_lang (from spec.lang)
- structured_fields.{job_title, seniority, company_size}
- classification.{seniority, company_size}
- fit_score.dimensions.{stage_match, headcount_match} via scoring.py
- expected_action hint from fit_band

Left null for labeller (TODO_LABEL):
- structured_fields.{industry, location, notable_signals}
- classification.{industry, segment}
- fit_score.dimensions.arr_match
- claims_allowed (verbatim source quotes)

Left null for labeller (TODO_RUBRIC):
- fit_score.dimensions.product_shape_match — anchor rubric (now resolved-gap version)
- fit_score.dimensions.role_match — anchor rubric

Left null for labeller (TODO_HOLISTIC):
- fit_score.value — labeller's holistic 0.0-1.0 call
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

from services.scoring import DEMO_ICP, headcount_match, stage_match

REPO = pathlib.Path(__file__).resolve().parent.parent
STAGING = REPO / "data" / "staging.jsonl"
OUT = REPO / "data" / "staging_with_gold.jsonl"

# Title-line preposition per language. Fallback to English form.
TITLE_PATTERNS = {
    "en": re.compile(r"^\s*(.+?)\s+at\s+(.+?)(?:\s*\([^)]*\))?\s*$"),
    "sv": re.compile(r"^\s*(.+?)\s+på\s+(.+?)(?:\s*\([^)]*\))?\s*$"),
    "de": re.compile(r"^\s*(.+?)\s+bei\s+(.+?)(?:\s*\([^)]*\))?\s*$"),
}

# Title → seniority bucket. First match wins; order matters (Founder before CTO).
SENIORITY_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:Co[\s\-]?Founder|Founder|Gründer|Grundare)\b", re.IGNORECASE), "Founder"),
    (re.compile(r"\b(?:CEO|CTO|CFO|CRO|CIO|Chief)\b", re.IGNORECASE), "C-level"),
    (re.compile(r"(?:^|\s)(?:VP|V\.?P\.?|Vice\sPresident)\b", re.IGNORECASE), "VP"),
    (re.compile(r"\bDirector\b|\bDirektör\b|\bDirektor\b", re.IGNORECASE), "Director"),
    (re.compile(r"\bHead\s+of\b", re.IGNORECASE), "Director"),
    (re.compile(r"\b(?:Senior\s+)?Manager\b|\bansvarig\b", re.IGNORECASE), "Manager"),
    (re.compile(r"\b(?:Senior\s+|Staff\s+|Lead\s+)?Engineer\b|\bIngenieur\b|\bIngenjör\b", re.IGNORECASE), "IC"),
    (re.compile(r"\b(?:Senior\s+)?(?:Product\s+)?Manager,?\s*AI\b", re.IGNORECASE), "Manager"),
    (re.compile(r"\bDesigner\b|\bConsultant\b|\bCoach\b", re.IGNORECASE), "IC"),
]

# Headcount: e.g. "~120 people", "approximately 80 employees", "rund 60 Mitarbeitende",
# "cirka 140 anställda", "around 280 people".
HEADCOUNT_PATTERNS = [
    re.compile(
        r"(?:~|approximately|approx\.|about|around|roughly|cirka|circa|rund|drygt|ungefär)\s*"
        r"(\d{1,4})\s*"
        r"(?:people|ppl|employees|engineers|anställda|Mitarbeitende|Mitarbeiter|Personen)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(\d{1,4})\s*(?:people|ppl|employees|anställda|Mitarbeitende|Mitarbeiter|Personen)",
        re.IGNORECASE,
    ),
]


def parse_title_line(title_line: str, lang: str) -> tuple[str | None, str | None]:
    pat = TITLE_PATTERNS.get(lang, TITLE_PATTERNS["en"])
    m = pat.match(title_line)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return (title_line.strip() or None), None


def derive_seniority(title: str | None) -> str | None:
    if not title:
        return None
    for pat, sen in SENIORITY_RULES:
        if pat.search(title):
            return sen
    return None


def extract_headcount(text: str) -> int | None:
    candidates: list[int] = []
    for pat in HEADCOUNT_PATTERNS:
        for m in pat.finditer(text):
            try:
                n = int(m.group(1))
            except ValueError:
                continue
            if 1 <= n <= 200_000:
                candidates.append(n)
    return max(candidates) if candidates else None


def company_size_bucket(n: int | None) -> str | None:
    if n is None:
        return None
    if n <= 10:
        return "1-10"
    if n <= 50:
        return "11-50"
    if n <= 200:
        return "51-200"
    if n <= 500:
        return "201-500"
    return "500+"


def split_paragraphs(text: str | None) -> list[str] | None:
    """Break a profile/company string into paragraph chunks for readable
    pretty-printed output. Drops blank separator lines; promote_gold rejoins
    with `\\n` between the first two entries (name + title) and `\\n\\n`
    between bio paragraphs."""
    if text is None:
        return None
    return [p for p in text.split("\n") if p != ""]


ACTION_HINT = {
    "strong": "auto_add",
    "ambiguous": "propose",
    "weak": "discard",
    "refuse": "refuse",
}


def scaffold(row: dict[str, Any]) -> dict[str, Any]:
    spec = row["spec"]
    profile: str = row["profile"]
    company: str = row.get("company") or ""
    full = f"{profile}\n\n{company}".strip()

    fit_band = spec["fit_band"]
    is_refuse = fit_band == "refuse"
    lang = spec.get("lang", "en")

    title_line = profile.split("\n", 2)[1] if "\n" in profile and len(profile.split("\n")) > 1 else ""
    job_title, _ = parse_title_line(title_line, lang)
    seniority = derive_seniority(job_title)
    if is_refuse and seniority is None:
        seniority = "IC"

    hc = extract_headcount(full)
    company_size = company_size_bucket(hc) if hc is not None else None

    spec_stage = spec.get("stage")
    stage_score: float | None
    if is_refuse:
        stage_score = 0.0
    elif spec_stage:
        stage_score = stage_match(spec_stage, DEMO_ICP)
    else:
        stage_score = None

    hc_score: float | None
    if is_refuse:
        hc_score = 0.0
    elif hc is not None:
        hc_score = headcount_match(hc, DEMO_ICP)
    else:
        hc_score = None

    placeholder_zero = 0.0 if is_refuse else None
    placeholder_value = 0.0 if is_refuse else None

    gold: dict[str, Any] = {
        "input_lang": lang,
        "structured_fields": {
            "job_title": job_title,
            "seniority": seniority,
            "company_size": "1-10" if is_refuse and company_size is None else company_size,
            "industry": "Insufficient signal" if is_refuse else None,
            "location": "not stated" if is_refuse else None,
            "notable_signals": ["Insufficient structural data to score"] if is_refuse else [],
        },
        "classification": {
            "industry": "Insufficient signal" if is_refuse else None,
            "segment": "Insufficient signal" if is_refuse else None,
            "seniority": seniority,
            "company_size": "1-10" if is_refuse and company_size is None else company_size,
        },
        "fit_score": {
            "value": placeholder_value,
            "dimensions": {
                "stage_match": stage_score,
                "headcount_match": hc_score,
                "arr_match": placeholder_zero,
                "product_shape_match": placeholder_zero,
                "role_match": placeholder_zero,
            },
        },
        "claims_allowed": [],
        "expected_action": ACTION_HINT[fit_band],
        "notes": (
            "Refuse: insufficient signal to judge against the ICP."
            if is_refuse
            else None
        ),
    }

    is_adversarial = row.get("kind") == "adversarial"
    if is_adversarial:
        gold["adversarial_pass_criteria"] = []

    todos: list[str] = []
    if not is_refuse:
        for path in (
            "structured_fields.industry",
            "structured_fields.location",
            "structured_fields.notable_signals",
            "classification.industry",
            "classification.segment",
            "fit_score.value",
            "fit_score.dimensions.arr_match",
            "fit_score.dimensions.product_shape_match",
            "fit_score.dimensions.role_match",
            "claims_allowed",
            "notes",
        ):
            todos.append(path)
        if stage_score is None:
            todos.append("fit_score.dimensions.stage_match (spec.stage missing)")
        if hc_score is None:
            todos.append("fit_score.dimensions.headcount_match (no headcount detected)")
    if is_adversarial:
        todos.append("adversarial_pass_criteria (boolean checks the model must pass)")

    return {
        "id": row["id"],
        "kind": row["kind"],
        "scenario": row["scenario"],
        "profile": split_paragraphs(profile),
        "company": split_paragraphs(row.get("company")),
        "spec": spec,
        "_extracted": {
            "job_title": job_title,
            "seniority": seniority,
            "headcount": hc,
            "company_size": company_size,
            "stage": spec_stage,
            "stage_match": stage_score,
            "headcount_match": hc_score,
        },
        "_todo": todos,
        "gold": gold,
    }


def main() -> int:
    rows = [json.loads(l) for l in STAGING.read_text().splitlines() if l.strip()]
    out_rows = [scaffold(r) for r in rows]
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, indent="\t") for r in out_rows) + "\n"
    )

    n_refuse = sum(1 for r in out_rows if r["spec"]["fit_band"] == "refuse")
    n_todos = sum(len(r["_todo"]) for r in out_rows)
    n_no_hc = sum(
        1 for r in out_rows
        if r["spec"]["fit_band"] != "refuse" and r["_extracted"]["headcount"] is None
    )
    print(f"wrote {len(out_rows)} rows to {OUT.relative_to(REPO)}")
    print(f"  refuse rows fully scaffolded: {n_refuse}")
    print(f"  total TODO fields across labelled rows: {n_todos}")
    if n_no_hc:
        print(f"  ⚠ {n_no_hc} non-refuse row(s) had no headcount detected — check regex coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
