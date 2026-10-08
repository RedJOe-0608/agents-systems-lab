BEGIN;

ALTER TABLE memories
    ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE',
    ADD COLUMN replaced_by_memory_id BIGINT
        REFERENCES memories(id);

ALTER TABLE memories
    ADD CONSTRAINT memories_status_check
        CHECK (
            status IN (
                'ACTIVE',
                'REDUNDANT',
                'MERGED_SOURCE',
                'SUPERSEDED'
            )
        ),
    ADD CONSTRAINT memories_replacement_not_self_check
        CHECK (
            replaced_by_memory_id IS NULL
            OR replaced_by_memory_id <> id
        ),
    ADD CONSTRAINT memories_status_replacement_check
        CHECK (
            (
                status = 'ACTIVE'
                AND replaced_by_memory_id IS NULL
            )
            OR
            (
                status IN (
                    'REDUNDANT',
                    'MERGED_SOURCE',
                    'SUPERSEDED'
                )
                AND replaced_by_memory_id IS NOT NULL
            )
        );

COMMIT;