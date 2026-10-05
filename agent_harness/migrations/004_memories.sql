CREATE TABLE memories (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    text TEXT NOT NULL,
    source_message_id BIGINT NOT NULL REFERENCES messages(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX memories_source_message_id_idx
    ON memories (source_message_id);