# lead-enrichment-eval

A B2B lead enrichment AI feature, built two ways. Same model (Claude Sonnet 4.6), different output contracts. The integrated build is structured, instrumented, and evaluated. The chat build is what an early-version B2B SaaS team realistically ships as their first AI feature: a well-written system prompt plus chat. Both run against the same eval set so the architectural difference is measurable.

## Thesis

A claim-grounded structured-output surface beats a free-form chat surface on extraction completeness, action accuracy, claim grounding, and adversarial robustness — at the cost of more upfront design and a tighter eval loop. This repo demonstrates that delta and exposes the eval scorecard.

## Architecture

- **Frontend (`web/`):** Next.js 15 App Router on Vercel. Routes: `/`, `/integrated`, `/chat`, `/scorecard`, `/methodology`, `/privacy`. API routes proxy to Modal.
- **Backend (`services/`):** Python on Modal, FastAPI app (`services/app.py`) exposing `/enrich`, `/chat` (SSE), `/neighbours`, `/healthz`. Anthropic SDK with prompt caching on both system prompts. The integrated build calls Sonnet 4.6 with extended thinking and the `enrich_lead` tool (strict JSON schema). The chat build streams free-form Sonnet 4.6 with no tools.
- **Database:** Postgres on Neon with `pgvector`. Schema in `migrations/0001_init.sql` — `requests` (telemetry, no input text written), `eval_set` (test items + Voyage-3 embeddings), `eval_runs` (nightly results).
- **Embeddings:** Voyage-3 for the eval-neighbour panel.
- **Rate limiting:** Upstash Redis sliding-window per IP.

## Project layout

```
services/        Modal Python services (FastAPI app, scoring, prompts, embeddings, telemetry)
tests/           pytest suite for the deterministic scoring functions
data/            Pre-loaded exemplars with full gold labels
migrations/      Postgres schema (idempotent)
web/             Next.js app
.github/         CI deploy workflow
```

## Local development

Python services:

```sh
uv venv --python 3.12 .venv
uv pip install -e .[dev]
.venv/bin/python -m pytest
```

Web app:

```sh
cd web && npm install && npm run dev
```

Copy `.env.example` to `.env` and fill in credentials. The web app needs `MODAL_BASE_URL`; the Python services need `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `DATABASE_URL` (Neon pooled, PgBouncer transaction mode), and `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN`.

To bring up the backend the first time: apply `migrations/0001_init.sql` against Neon, `modal secret create lead-enrichment` with the API keys, `modal deploy services/app.py`, set `MODAL_BASE_URL` on Vercel, and `python -m services.eval_seed` to populate `eval_set` with embedded exemplars.

## License

MIT. See `LICENSE`.
