BEGIN;

ALTER TABLE memories
    ADD COLUMN entities JSONB NOT NULL DEFAULT '[]'::jsonb;

DROP TABLE memory_entities;
DROP TABLE entity_mentions;
DROP TABLE entities;

COMMIT;