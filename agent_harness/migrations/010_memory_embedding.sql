ALTER TABLE memories
    ADD COLUMN embedding halfvec(2048);

CREATE INDEX memories_embedding_hnsw_idx
    ON memories USING hnsw (embedding halfvec_cosine_ops);