"""
Voyage-3 embeddings for the eval-neighbour panel.

Live calls embed incoming requests. The eval set is precomputed and stored on
`eval_set.embedding`; live results are stored on `requests.embedding`.
"""

from __future__ import annotations

import os

import voyageai

EMBED_MODEL = "voyage-3"
EMBED_DIM = 1024

_client: voyageai.AsyncClient | None = None


def _get_client() -> voyageai.AsyncClient:
    global _client
    if _client is None:
        api_key = os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY is not set")
        _client = voyageai.AsyncClient(api_key=api_key)
    return _client


async def embed(text: str, *, input_type: str = "document") -> list[float]:
    client = _get_client()
    result = await client.embed([text], model=EMBED_MODEL, input_type=input_type)
    return result.embeddings[0]


async def find_neighbours(text: str, k: int = 3) -> list[dict]:
    """Return up to k nearest test-set items by cosine similarity."""
    from services.db import connection

    vec = await embed(text, input_type="query")
    async with connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                """
                SELECT id, gold->'fit_score'->>'value' AS score,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM eval_set
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec, vec, k),
            )
            rows = await cur.fetchall()
    return [
        {"id": row[0], "score": float(row[1]) if row[1] is not None else None,
         "similarity": float(row[2])}
        for row in rows
    ]
