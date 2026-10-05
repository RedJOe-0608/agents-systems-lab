CREATE TABLE memory_edges (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_memory_id BIGINT NOT NULL REFERENCES memories(id),
    target_memory_id BIGINT NOT NULL REFERENCES memories(id),
    relation_type TEXT NOT NULL CHECK (
        relation_type IN ('RELATED_TO', 'SHARED_ENTITY', 'CAUSES')
    ),
    score REAL NOT NULL CHECK (score BETWEEN 0 AND 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CHECK (source_memory_id <> target_memory_id),
    CHECK (
        relation_type = 'CAUSES'
        OR source_memory_id < target_memory_id
    ),
    UNIQUE (source_memory_id, target_memory_id, relation_type)
);

CREATE INDEX memory_edges_target_idx
    ON memory_edges (target_memory_id);