-- Snapshot-served flag for the requests table.
-- Distinguishes rows that were served from a committed exemplar JSON snapshot
-- (no Anthropic call) from rows that hit the live API. Separate from
-- `cache_hit`, which means "Anthropic prompt-cache hit." Cost rollups should
-- filter out `snapshot_served = true` rows so the daily-cost tripwire is not
-- inflated by snapshot replays.

ALTER TABLE requests
  ADD COLUMN IF NOT EXISTS snapshot_served bool NOT NULL DEFAULT false;
