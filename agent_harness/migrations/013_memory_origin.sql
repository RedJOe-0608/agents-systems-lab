BEGIN;

ALTER TABLE memories
    ADD COLUMN origin_type TEXT NOT NULL DEFAULT 'EXTRACTED';

ALTER TABLE memories
    ALTER COLUMN source_message_id DROP NOT NULL;

ALTER TABLE memories
    ADD CONSTRAINT memories_origin_type_check
        CHECK (
            origin_type IN ('EXTRACTED', 'MERGED')
        ),
    ADD CONSTRAINT memories_origin_source_check
        CHECK (
            (
                origin_type = 'EXTRACTED'
                AND source_message_id IS NOT NULL
            )
            OR
            (
                origin_type = 'MERGED'
                AND source_message_id IS NULL
            )
        );

COMMIT;