CREATE TABLE memory_entities (
    memory_id BIGINT NOT NULL REFERENCES memories(id),
    entity_id BIGINT NOT NULL REFERENCES entities(id),
    PRIMARY KEY (memory_id, entity_id)
);

CREATE INDEX memory_entities_entity_id_idx
    ON memory_entities (entity_id);