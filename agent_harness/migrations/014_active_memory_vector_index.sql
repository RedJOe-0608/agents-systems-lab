BEGIN;

DROP INDEX memories_embedding_hnsw_idx;

CREATE INDEX memories_active_embedding_hnsw_idx
    ON memories
    USING hnsw (embedding halfvec_cosine_ops)
    WHERE status = 'ACTIVE'
      AND embedding IS NOT NULL;

COMMIT;