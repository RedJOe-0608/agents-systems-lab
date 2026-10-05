CREATE TABLE entities (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name TEXT NOT NULL,
    known_aliases TEXT[] NOT NULL DEFAULT '{}',
    type TEXT NOT NULL,
    context TEXT
);