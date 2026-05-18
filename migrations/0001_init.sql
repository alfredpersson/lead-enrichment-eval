-- Phase 1 schema. Apply once against the Neon database before first deploy.
-- Voyage-3 produces 1024-dim embeddings; pgvector handles cosine search.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- One row per inference call (both modes). Raw input text is never written.
CREATE TABLE IF NOT EXISTS requests (
    request_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    ts                timestamptz NOT NULL DEFAULT now(),
    mode              text NOT NULL CHECK (mode IN ('integrated', 'chat')),
    example_id        text,
    input_lang        text,
    input_char_count  int,
    model_id          text,
    thinking_enabled  bool,
    input_tokens      int,
    output_tokens     int,
    thinking_tokens   int,
    cache_hit         bool,
    latency_ms        int,
    embedding         vector(1024),
    error             text,

    -- integrated mode
    action                            text,
    fit_score                         real,
    claim_count                       int,
    claims_with_source_quote_count    int,

    -- chat mode
    turn_count          int,
    extractor_complete  bool
);

CREATE INDEX IF NOT EXISTS requests_ts_idx ON requests (ts DESC);
CREATE INDEX IF NOT EXISTS requests_mode_ts_idx ON requests (mode, ts DESC);

-- Test set; one row per labelled item, with precomputed embedding for neighbour lookup.
CREATE TABLE IF NOT EXISTS eval_set (
    id            text PRIMARY KEY,
    version       text NOT NULL,
    scenario      text,
    profile       text NOT NULL,
    company       text,
    gold          jsonb NOT NULL,
    embedding     vector(1024),
    created_at    timestamptz NOT NULL DEFAULT now()
);

-- No ANN index on eval_set.embedding: at v1's 73 rows a sequential scan over
-- 1024-dim cosine is exact and sub-millisecond. IVFFlat needs ~1000+ rows per
-- list to deliver useful recall, so a small-N index is strictly worse than no
-- index. Revisit when the test set crosses ~5K rows.

-- Nightly eval results, one row per (run, mode).
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    ts              timestamptz NOT NULL DEFAULT now(),
    test_set_sha    text NOT NULL,
    test_set_version text NOT NULL,
    mode            text NOT NULL CHECK (mode IN ('integrated', 'chat')),
    metrics         jsonb NOT NULL,
    notes           text
);

CREATE INDEX IF NOT EXISTS eval_runs_ts_idx ON eval_runs (ts DESC);
