-- Phase 0b prereq: kind discriminator on the eval set.
-- Values: 'exemplar' (the 5 hand-crafted demo cards), 'synthetic' (Phase 0b
-- bulk-generated profiles), 'adversarial' (injection / jailbreak / contradictory
-- / sparse / multilingual_injection), 'edge' (very_short / very_long /
-- ambiguous_title / non_english_supported). Sub-kinds for adversarial and edge
-- live inside the gold jsonb.

ALTER TABLE eval_set ADD COLUMN IF NOT EXISTS kind text NOT NULL DEFAULT 'exemplar';

CREATE INDEX IF NOT EXISTS eval_set_kind_idx ON eval_set (kind);
