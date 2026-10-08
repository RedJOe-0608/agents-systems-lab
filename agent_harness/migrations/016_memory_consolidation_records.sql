BEGIN;

CREATE TABLE memory_consolidation_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    decision_version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'RUNNING' CHECK (
        status IN ('RUNNING', 'COMPLETED', 'FAILED')
    ),
    batch_limit INTEGER NOT NULL CHECK (batch_limit > 0),
    candidate_limit INTEGER NOT NULL CHECK (candidate_limit > 0),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ,
    seed_count INTEGER NOT NULL DEFAULT 0,
    proposed_pair_count INTEGER NOT NULL DEFAULT 0,
    evaluated_pair_count INTEGER NOT NULL DEFAULT 0,
    applied_action_count INTEGER NOT NULL DEFAULT 0,
    error_message TEXT
);

CREATE TABLE memory_consolidation_run_memories (
    run_id BIGINT NOT NULL
        REFERENCES memory_consolidation_runs(id),
    memory_id BIGINT NOT NULL REFERENCES memories(id),
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (
        status IN ('PENDING', 'COMPLETED', 'FAILED')
    ),
    processed_at TIMESTAMPTZ,
    error_message TEXT,
    PRIMARY KEY (run_id, memory_id)
);

CREATE INDEX memory_consolidation_run_memories_memory_idx
    ON memory_consolidation_run_memories (memory_id);

CREATE TABLE memory_consolidation_decisions (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id BIGINT REFERENCES memory_consolidation_runs(id),
    memory_a_id BIGINT NOT NULL REFERENCES memories(id),
    memory_b_id BIGINT NOT NULL REFERENCES memories(id),
    current_memory_id BIGINT NOT NULL REFERENCES memories(id),
    candidate_memory_id BIGINT NOT NULL REFERENCES memories(id),

    redundancy_score REAL NOT NULL CHECK (
        redundancy_score BETWEEN 0 AND 1
    ),
    contradiction_score REAL NOT NULL CHECK (
        contradiction_score BETWEEN 0 AND 1
    ),
    current_supersedes_candidate_score REAL NOT NULL CHECK (
        current_supersedes_candidate_score BETWEEN 0 AND 1
    ),
    candidate_supersedes_current_score REAL NOT NULL CHECK (
        candidate_supersedes_current_score BETWEEN 0 AND 1
    ),

    representation TEXT NOT NULL CHECK (
        representation IN ('keep_separate', 'merge', 'uncertain')
    ),
    representation_probability REAL NOT NULL CHECK (
        representation_probability BETWEEN 0 AND 1
    ),
    representation_probabilities JSONB NOT NULL,

    decision_version TEXT NOT NULL,
    model_id TEXT NOT NULL,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    applied_action TEXT NOT NULL DEFAULT 'PENDING' CHECK (
        applied_action IN (
            'PENDING',
            'NONE',
            'CANONICALIZED',
            'MERGED',
            'SUPERSEDED',
            'CONTRADICTION_LINKED',
            'DEFERRED'
        )
    ),
    result_memory_id BIGINT REFERENCES memories(id),
    applied_at TIMESTAMPTZ,
    action_error TEXT,

    CHECK (memory_a_id < memory_b_id),
    CHECK (
        current_memory_id IN (memory_a_id, memory_b_id)
        AND candidate_memory_id IN (memory_a_id, memory_b_id)
        AND current_memory_id <> candidate_memory_id
    ),
    UNIQUE (memory_a_id, memory_b_id, decision_version)
);

CREATE INDEX memory_consolidation_decisions_run_idx
    ON memory_consolidation_decisions (run_id);

CREATE INDEX memory_consolidation_decisions_pending_idx
    ON memory_consolidation_decisions (decision_version, id)
    WHERE applied_action = 'PENDING';

CREATE TABLE memory_lifecycle_edges (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_memory_id BIGINT NOT NULL REFERENCES memories(id),
    target_memory_id BIGINT NOT NULL REFERENCES memories(id),
    relation_type TEXT NOT NULL CHECK (
        relation_type IN (
            'REDUNDANT_OF',
            'MERGED_INTO',
            'SUPERSEDES',
            'CONTRADICTS'
        )
    ),
    score REAL NOT NULL CHECK (score BETWEEN 0 AND 1),
    decision_id BIGINT NOT NULL
        REFERENCES memory_consolidation_decisions(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

    CHECK (source_memory_id <> target_memory_id),
    CHECK (
        relation_type <> 'CONTRADICTS'
        OR source_memory_id < target_memory_id
    ),
    UNIQUE (
        source_memory_id,
        target_memory_id,
        relation_type
    )
);

CREATE INDEX memory_lifecycle_edges_target_idx
    ON memory_lifecycle_edges (target_memory_id);

CREATE INDEX memories_active_consolidation_idx
    ON memories (
        last_consolidated_at NULLS FIRST,
        created_at,
        id
    )
    WHERE status = 'ACTIVE';

COMMIT;
