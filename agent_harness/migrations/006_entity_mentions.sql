CREATE TABLE entity_mentions (
    entity_id BIGINT NOT NULL REFERENCES entities(id),
    message_id BIGINT NOT NULL REFERENCES messages(id),
    PRIMARY KEY (entity_id, message_id)
);

CREATE INDEX entity_mentions_message_id_idx
    ON entity_mentions (message_id);