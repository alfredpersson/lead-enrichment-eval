"""
Seed `eval_set` from `data/exemplars.json` and precompute Voyage-3 embeddings.

Run locally (or as a one-off Modal function) after applying migrations and
before the first eval run. Idempotent: rows are upserted by id.

Test-set schema (extended for Phase 0b):

    {
      "version": str,           # bumped to "v1.0" once bulk labelling lands
      "items": [
        {
          "id": str,
          "kind": "exemplar" | "synthetic" | "adversarial" | "edge",
          "scenario": str,      # human-readable subtype label
          "label": str,         # short title for tooling
          "profile": str,
          "company": str | null,
          "gold": {
            ...                 # full gold-label payload (see exemplars 1-5)
            "adversarial_kind"?: str,   # only when kind == "adversarial"
            "edge_kind"?: str,          # only when kind == "edge"
            "adversarial_pass_criteria"?: list[str],  # adversarial only
          }
        },
        ...
      ]
    }
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys

from services.config import is_local
from services.db import connection
from services.embeddings import embed

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
EXEMPLARS_PATH = REPO_ROOT / "data" / "exemplars.json"


async def seed(path: pathlib.Path = EXEMPLARS_PATH) -> int:
    payload = json.loads(path.read_text())
    version = payload.get("version", "0a")
    items = payload["items"]

    async with connection() as conn:
        async with conn.cursor() as cur:
            for item in items:
                text = item["profile"]
                if item.get("company"):
                    text = f"{text}\n\n{item['company']}"
                vec = await embed(text)
                await cur.execute(
                    """
                    INSERT INTO eval_set (id, version, kind, scenario, profile, company, gold, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                    ON CONFLICT (id) DO UPDATE SET
                        version = EXCLUDED.version,
                        kind = EXCLUDED.kind,
                        scenario = EXCLUDED.scenario,
                        profile = EXCLUDED.profile,
                        company = EXCLUDED.company,
                        gold = EXCLUDED.gold,
                        embedding = EXCLUDED.embedding
                    """,
                    (
                        item["id"],
                        version,
                        item.get("kind", "exemplar"),
                        item.get("scenario"),
                        item["profile"],
                        item.get("company"),
                        json.dumps(item["gold"]),
                        vec,
                    ),
                )
        await conn.commit()
    return len(items)


def main() -> int:
    if is_local():
        print("APP_ENV=local: refusing to seed (no DB). Set APP_ENV=prod to run.")
        return 1
    n = asyncio.run(seed())
    print(f"Seeded {n} eval-set rows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
