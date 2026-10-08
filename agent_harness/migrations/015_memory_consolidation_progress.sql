BEGIN;

ALTER TABLE memories
    ADD COLUMN last_consolidated_version TEXT,
    ADD COLUMN last_consolidated_at TIMESTAMPTZ;

COMMIT;