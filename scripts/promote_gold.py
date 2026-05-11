"""
Move fully-labelled rows from staging_with_gold.jsonl into data/exemplars.json.

A row counts as "fully labelled" when:
- expected_action == 'refuse' (refuse rows scaffold complete on the first pass), OR
- structured_fields.{industry, location, notable_signals} are non-null/non-empty
- classification.{industry, segment} are non-null
- fit_score.value is non-null and every dimension is non-null
- claims_allowed has at least one entry

Usage:

    python -m scripts.promote_gold             # dry run; lists what's ready vs blocked
    python -m scripts.promote_gold --commit    # actually update data/exemplars.json

The promote step is idempotent: rows whose id is already in exemplars.json are
skipped. Run repeatedly as you label more rows.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

REPO = pathlib.Path(__file__).resolve().parent.parent
STAGING = REPO / "data" / "staging_with_gold.jsonl"
EXEMPLARS = REPO / "data" / "exemplars.json"


def is_complete(gold: dict[str, Any]) -> tuple[bool, list[str]]:
    if gold.get("expected_action") == "refuse":
        return True, []
    issues: list[str] = []
    if gold.get("fit_score", {}).get("value") is None:
        issues.append("fit_score.value")
    dims = gold.get("fit_score", {}).get("dimensions", {})
    for k in ("stage_match", "headcount_match", "arr_match", "product_shape_match", "role_match"):
        if dims.get(k) is None:
            issues.append(f"fit_score.dimensions.{k}")
    sf = gold.get("structured_fields", {})
    for k in ("industry", "location"):
        if sf.get(k) is None:
            issues.append(f"structured_fields.{k}")
    if not sf.get("notable_signals"):
        issues.append("structured_fields.notable_signals")
    cl = gold.get("classification", {})
    for k in ("industry", "segment"):
        if cl.get(k) is None:
            issues.append(f"classification.{k}")
    if not gold.get("claims_allowed"):
        issues.append("claims_allowed")
    return (not issues), issues


def humanise_scenario(s: str) -> str:
    parts = s.replace("_", " ").split()
    return " ".join(p.capitalize() if p.isalpha() and not p.isupper() else p for p in parts)


def join_paragraphs(value: Any) -> Any:
    """Staging stores profile/company as arrays of paragraphs for readability;
    exemplars.json expects a single string with the title line on its own line
    and bio paragraphs separated by blank lines."""
    if not isinstance(value, list):
        return value
    if len(value) <= 1:
        return value[0] if value else ""
    return f"{value[0]}\n{value[1]}" + (
        "\n\n" + "\n\n".join(value[2:]) if len(value) > 2 else ""
    )


def to_exemplar(row: dict[str, Any]) -> dict[str, Any]:
    label = row.get("label") or humanise_scenario(row["scenario"])
    return {
        "id": row["id"],
        "kind": row["kind"],
        "scenario": row["scenario"],
        "label": label,
        "profile": join_paragraphs(row["profile"]),
        "company": join_paragraphs(row.get("company")),
        "gold": row["gold"],
    }


def load_staging(text: str) -> list[dict[str, Any]]:
    """Parse staging file as a sequence of pretty-printed JSON objects.
    (Not standard JSONL — objects span multiple lines for human editing.)"""
    dec = json.JSONDecoder()
    rows: list[dict[str, Any]] = []
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            break
        obj, end = dec.raw_decode(text, i)
        rows.append(obj)
        i = end
    return rows


def sort_key(it: dict[str, Any]) -> tuple[int, str]:
    rid = it["id"]
    return (int(rid), "") if rid.isdigit() else (99_999, rid)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    ap.add_argument("--commit", action="store_true", help="Actually write to exemplars.json")
    args = ap.parse_args()

    rows = load_staging(STAGING.read_text())
    exemplars = json.loads(EXEMPLARS.read_text())
    existing_ids = {it["id"] for it in exemplars["items"]}

    ready: list[dict[str, Any]] = []
    blocked: list[tuple[str, list[str]]] = []
    skipped_already_in = 0
    for row in rows:
        if row["id"] in existing_ids:
            skipped_already_in += 1
            continue
        ok, issues = is_complete(row.get("gold", {}))
        if ok:
            ready.append(to_exemplar(row))
        else:
            blocked.append((row["id"], issues))

    print(f"already in exemplars.json:   {skipped_already_in}")
    print(f"ready to promote:            {len(ready)}")
    print(f"blocked (incomplete labels): {len(blocked)}")
    for rid, issues in blocked[:8]:
        head = ", ".join(issues[:3])
        more = f" (+{len(issues) - 3} more)" if len(issues) > 3 else ""
        print(f"  id={rid}: {head}{more}")
    if len(blocked) > 8:
        print(f"  ... and {len(blocked) - 8} more")

    if not args.commit:
        print("\n(dry run; pass --commit to write)")
        return 0

    if not ready:
        print("nothing to commit")
        return 0

    exemplars["items"].extend(ready)
    exemplars["items"].sort(key=sort_key)
    EXEMPLARS.write_text(json.dumps(exemplars, indent=2, ensure_ascii=False) + "\n")
    print(f"committed {len(ready)} new row(s) to {EXEMPLARS.relative_to(REPO)}")
    print(f"exemplars.json now contains {len(exemplars['items'])} rows total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
