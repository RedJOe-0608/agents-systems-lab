# Memory Consolidation Plan

Status: core implementation complete
Last updated: 2026-10-08

This document records the design and the implemented consolidation subsystem
for the current PostgreSQL memory implementation. Production scheduling,
calibrated thresholds, and cost/token budgeting remain deferred.

The implementation is split across:

- migrations `012` through `016` for lifecycle, origin, active indexes,
  scheduling progress, run records, pair decisions, and lifecycle edges;
- `agent_harness/memory_consolidation.py` for pair discovery, generated-merge
  preparation, and batch orchestration;
- `agent_harness/memory_lifecycle.py` for action selection and transactional
  lifecycle mutations;
- `agent_harness/storage/` for connections, conversations, memory queries,
  consolidation runs, decisions, and lifecycle reads;
- `agent_harness/memory_retrieval.py` for contradiction counterpart retrieval;
- `run_memory_consolidation.py` for manual invocation.

## 1. Current memory system

The current write path is:

```text
session ends
  -> extract atomic memories with one source message each
  -> embed each memory
  -> find at most 10 write candidates
  -> ask Jev for semantic, causal, and shared-entity probabilities
  -> store the memory
  -> store RELATED_TO, SHARED_ENTITY, and CAUSES edges
```

The current read path is:

```text
query
  -> Jev graph routing and multi-hop need
  -> vector + BM25 anchors fused with RRF
  -> Jev evidence assessment
  -> bounded semantic/entity/causal graph expansion
  -> repeated evidence assessment
  -> ephemeral memory context for the answering model
```

Relevant tables:

- `memories`: one row per extracted or generated memory, a nullable direct
  `source_message_id`, lifecycle state, entities JSONB, and an optional
  `halfvec(2048)` embedding.
- `memory_edges`: knowledge edges only: `RELATED_TO`, `SHARED_ENTITY`, and directed `CAUSES`.
- `memory_lifecycle_edges`: consolidation lineage and contradiction edges.
- `memory_consolidation_decisions`: versioned, unordered pair decisions.
- `messages` and `sessions`: immutable source conversation records.

The original limitations addressed by this implementation were:

- All memories are treated as equally active.
- A memory can reference only one direct source message.
- Search queries do not filter by lifecycle status.
- A derived merged memory cannot be represented honestly because it has no single direct source message.
- There is no durable record of which memory pairs have already been evaluated for consolidation.
- There is no distinction between knowledge edges and memory-lifecycle relationships.

## 2. Goals

Consolidation should create a smaller, cleaner active-memory view without deleting evidence.

The first version will handle four outcomes:

1. **Redundancy**: two memories state the same fact without unique details.
2. **Merge**: compatible memories about the same fact contain different explicit details that can be combined without loss.
3. **Supersession**: one memory explicitly replaces a previously valid fact.
4. **Contradiction**: two memories make incompatible claims that cannot both be accepted.

The following are deliberately out of scope:

- promotion into general behavioral patterns;
- temporal graph construction;
- deleting raw memories;
- comparing every memory with every other memory;
- using consolidation relationships as additional Jev-routed graph types;
- retry, token-budget, and production scheduling hardening until the core path works.

## 3. Locked behavior

### 3.1 Background and bounded

Consolidation will run outside the interactive write/read critical path. A run selects a bounded batch of active seed memories. For each seed it discovers only a bounded Top-K candidate set through the existing hybrid candidate signals.

```text
Not: every memory x every memory

Instead:
selected active seeds x Top-K candidates
```

The implementation exposes a manually invoked batch function and CLI. A
scheduler or worker can call the same function later.

### 3.2 Pair-based tracking

Evaluation is tracked per unordered memory pair, not with a single permanent `memory_was_evaluated` flag.

For IDs `8` and `3`, the canonical pair is always:

```text
(3, 8)
```

If memory 3 is evaluated against memory 8, processing memory 8 later must not evaluate `(8, 3)` again under the same decision version.

A memory-level `last_consolidated_at` field is only a scheduling hint. The pair-decision table is the source of truth for whether a pair has been evaluated.

### 3.3 Raw evidence survives

No consolidation action physically deletes a memory or source message.

Inactive memories remain available for:

- provenance;
- audits;
- debugging;
- reconsideration under a later consolidation policy;
- recovery if a generated merged memory is found to be poor.

### 3.4 Active-memory search

Normal vector search, BM25 search, entity candidate search, write candidate search, and graph-neighbor expansion will use only `ACTIVE` memories.

Lifecycle relationships are resolved separately from semantic, entity, and causal traversal.

### 3.5 No promotion

Promotion is deferred. The representation Choice will contain only:

- `keep_separate`;
- `merge`;
- `uncertain`.

## 4. Jev consolidation decision

Each unevaluated pair is sent to Jev with a shared state containing two complete memory objects.

```json
{
  "current_memory": {
    "memory_id": 12,
    "content": "The user's memory system uses PostgreSQL.",
    "entities": [],
    "created_at": "..."
  },
  "candidate_memory": {
    "memory_id": 4,
    "content": "The user's memory backend is Postgres.",
    "entities": [],
    "created_at": "..."
  }
}
```

`current_memory` is the seed being examined and `candidate_memory` is the
retrieved candidate. Insertion order does not establish factual recency, so
supersession is evaluated explicitly in both directions.

### 4.1 Independent Noul questions

#### Redundancy

Does one memory repeat the same fact as the other without adding a recallable detail?

- True: duplicate or paraphrase of the same claim with no unique detail.
- False: different facts, distinct events, or either memory adds meaningful detail.

#### Contradiction

Do the memories make incompatible claims about the same subject under compatible context?

- True: both claims cannot be accepted simultaneously.
- False: compatible details, uncertainty, different subjects, or an explicit update explains the difference.

#### Supersession in both directions

Does `current_memory` explicitly replace or correct a previously valid fact in `candidate_memory`?

Does `candidate_memory` explicitly replace or correct a previously valid fact in `current_memory`?

- True: the newer claim is an explicit update or correction.
- False: mere recency, different wording, or an unrelated fact.

### 4.2 Representation Choice

#### `keep_separate`

The memories are distinct facts, contradictory accounts, or contain details that should remain independently represented.

#### `merge`

The memories describe the same fact or event, contain compatible explicit details, and can be combined without losing information.

#### `uncertain`

The supplied evidence is insufficient to safely keep, merge, or otherwise resolve the pair automatically.

### 4.3 No useful-link question

The write path already evaluates semantic, entity, and causal relationships. Consolidation will not repeat a generic useful-link decision. `RELATED_TO`, `SHARED_ENTITY`, and `CAUSES` remain knowledge-graph concerns.

## 5. Action policy

Initial automatic threshold proposal: `0.85`, matching the reference Jev-Mem consolidation profile. This value must later be evaluated rather than treated as calibrated probability.

Apply results in the following safety order:

1. High contradiction blocks redundancy, supersession, and merge actions.
2. High explicit supersession marks the older memory superseded.
3. High redundancy canonicalizes the pair without generation.
4. A high-confidence `merge` Choice may authorize a generative merge.
5. `keep_separate`, `uncertain`, or weak scores perform no lifecycle mutation.

### 5.1 Redundancy

Example:

```text
Memory 1: The user's memory system uses PostgreSQL.
Memory 3: Postgres is the backend of the user's memory system.
```

Action:

- choose one existing memory as canonical;
- leave the canonical memory `ACTIVE`;
- mark the other memory `REDUNDANT`;
- set the inactive memory's replacement pointer to the canonical memory;
- create an audit lifecycle relationship;
- preserve both original source messages;
- transfer/upsert useful knowledge edges from the inactive duplicate to the canonical memory.

No third memory and no generative call are needed.

The initial canonical-selection rule should be deterministic. Prefer the already-established/older active memory unless a later policy explicitly scores representation quality. Stable canonical IDs avoid unnecessary graph rewiring.

### 5.2 Merge

Example:

```text
Memory 1: The memory system uses PostgreSQL.
Memory 3: The memory system stores embeddings through pgvector.
```

Possible merged memory:

```text
The memory system uses PostgreSQL with pgvector for embedding storage.
```

Action:

- generate one atomic text containing only explicitly supported compatible details;
- validate nonempty structured output;
- embed and save it as an `ACTIVE`, `MERGED`-origin memory;
- run normal write-time relation construction for the new memory;
- create `MERGED_INTO` lifecycle relationships from both sources to the result;
- mark both source memories `MERGED_SOURCE` and point them to the result;
- expire sources only after the merged memory and its required relationships are safely stored.

The merge generator must not infer a general preference or pattern. That would be promotion and is out of scope.

### 5.3 Supersession

Example:

```text
Old: The user's memory backend is SQLite.
New: The user migrated the memory backend from SQLite to PostgreSQL.
```

Action:

- keep the explicit replacement `ACTIVE`;
- mark the old memory `SUPERSEDED`;
- point the old memory to its active replacement;
- create a directed `SUPERSEDES` lifecycle relationship from new to old;
- do not delete the old memory.

Because temporal retrieval is out of scope, normal search returns only the active replacement. Historical provenance remains queryable through lifecycle records and source messages.

### 5.4 Contradiction

Example:

```text
Memory 1: The user's memory backend is PostgreSQL.
Memory 3: The user's memory backend is SQLite.
```

Action:

- keep both memories `ACTIVE`;
- create one symmetric `CONTRADICTS` lifecycle relationship;
- when either memory enters retrieval evidence, deterministically add the contradictory counterpart;
- allow evidence assessment and the answering model to see the unresolved conflict.

## 6. Implemented database schema

The schema was split into migrations `012` through `016` so each concept could
be reviewed and applied independently. The snippets below summarize the
resulting shape; the migration files are the executable source of truth.

### 6.1 Extend `memories`

```sql
ALTER TABLE memories
    ADD COLUMN status TEXT NOT NULL DEFAULT 'ACTIVE'
        CHECK (
            status IN (
                'ACTIVE',
                'REDUNDANT',
                'MERGED_SOURCE',
                'SUPERSEDED'
            )
        ),
    ADD COLUMN origin_type TEXT NOT NULL DEFAULT 'EXTRACTED'
        CHECK (origin_type IN ('EXTRACTED', 'MERGED')),
    ADD COLUMN replaced_by_memory_id BIGINT
        REFERENCES memories(id),
    ADD COLUMN last_consolidated_at TIMESTAMPTZ,
    ADD COLUMN last_consolidated_version TEXT,
    ADD CHECK (
        replaced_by_memory_id IS NULL
        OR replaced_by_memory_id <> id
    ),
    ADD CHECK (
        (
            status = 'ACTIVE'
            AND replaced_by_memory_id IS NULL
        )
        OR (
            status IN (
                'REDUNDANT',
                'MERGED_SOURCE',
                'SUPERSEDED'
            )
            AND replaced_by_memory_id IS NOT NULL
        )
    );

ALTER TABLE memories
    ALTER COLUMN source_message_id DROP NOT NULL;

CREATE INDEX memories_active_created_idx
    ON memories (created_at, id)
    WHERE status = 'ACTIVE';
```

Why `source_message_id` becomes nullable:

- extracted memories still have one direct source message;
- generated merged memories are supported by source memories, not by one specific message;
- their message provenance is reached transitively through the source memories.

Why `replaced_by_memory_id` exists:

- it provides fast resolution from an inactive memory to its active canonical/replacement memory;
- the lifecycle-edge table remains the richer audit trail;
- both must be updated in one transaction.

### 6.2 Replace the vector index with an active partial index

The current HNSW index contains every memory. Default retrieval should index only active memories.

```sql
DROP INDEX memories_embedding_hnsw_idx;

CREATE INDEX memories_active_embedding_hnsw_idx
    ON memories
    USING hnsw (embedding halfvec_cosine_ops)
    WHERE status = 'ACTIVE'
      AND embedding IS NOT NULL;
```

Inactive memories remain directly addressable by ID for provenance and audits.

### 6.3 Consolidation runs

```sql
CREATE TABLE memory_consolidation_runs (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    decision_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
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
```

This table records each background/manual batch and supports later crash recovery and observability.

### 6.4 Seed memories in each run

```sql
CREATE TABLE memory_consolidation_run_memories (
    run_id BIGINT NOT NULL
        REFERENCES memory_consolidation_runs(id),
    memory_id BIGINT NOT NULL
        REFERENCES memories(id),
    status TEXT NOT NULL DEFAULT 'PENDING' CHECK (
        status IN ('PENDING', 'COMPLETED', 'FAILED')
    ),
    processed_at TIMESTAMPTZ,
    error_message TEXT,
    PRIMARY KEY (run_id, memory_id)
);

CREATE INDEX memory_consolidation_run_memories_memory_idx
    ON memory_consolidation_run_memories (memory_id);
```

### 6.5 Pair decisions

```sql
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
        representation IN (
            'keep_separate',
            'merge',
            'uncertain'
        )
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
        current_memory_id = memory_a_id
        OR current_memory_id = memory_b_id
    ),
    UNIQUE (memory_a_id, memory_b_id, decision_version)
);

CREATE INDEX memory_consolidation_decisions_run_idx
    ON memory_consolidation_decisions (run_id);
```

`memory_a_id` and `memory_b_id` always use ascending order.
`current_memory_id` and `candidate_memory_id` preserve both roles used by the
two directional supersession questions.

`representation_probabilities` stores the full Choice distribution. Automatic merge decisions must use the selected option's probability, not a separate generic confidence value.

### 6.6 Lifecycle relationships

Knowledge edges remain in `memory_edges`. Consolidation relationships use a separate table so they are not accidentally traversed as semantic, causal, or entity paths.

```sql
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
```

Directions:

- `REDUNDANT_OF`: inactive duplicate -> active canonical memory.
- `MERGED_INTO`: inactive source -> active merged result.
- `SUPERSEDES`: active replacement -> inactive previous memory.
- `CONTRADICTS`: symmetric; store lower ID as source.

This table is also the normalized equivalent of `source_memory_ids`. Do not store source-memory arrays on `memories`; query incoming `REDUNDANT_OF` or `MERGED_INTO` relationships instead.

## 7. Implemented code changes

### 7.1 Database reads

The following functions now default to `m.status = 'ACTIVE'`:

- `load_memories_with_sources`;
- `search_vector_candidates`;
- `find_entity_candidate_ids`;
- `load_candidates_by_ids` when used for normal retrieval/candidate discovery;
- `search_bm25_candidates` through its base loader;
- `find_write_candidates` through the above functions;
- `find_query_anchors` through the above functions;
- `load_graph_neighbors` for the neighbor row.

Separate audit/provenance loaders must be able to read inactive memories explicitly.

### 7.2 Source joins

The current read functions use an inner join:

```sql
JOIN messages AS msg ON msg.id = m.source_message_id
```

After merged memories are possible, this must become a left join in generic memory loaders:

```sql
LEFT JOIN messages AS msg ON msg.id = m.source_message_id
```

Callers must accept null `role` and `payload` for derived merged memories.

### 7.3 Saving memories

Keep `save_memory` for directly extracted memories. Add a separate derived-memory function rather than overloading its meaning:

```text
save_merged_memory(text, entities, embedding)
```

It inserts:

```text
source_message_id = NULL
origin_type = MERGED
status = ACTIVE
```

Lineage is then recorded with `MERGED_INTO` lifecycle rows from the source memories.

### 7.4 Knowledge-edge migration during canonicalization

When memory B becomes `REDUNDANT_OF` memory A, useful knowledge edges attached to B must not disappear from active traversal.

Within the same transaction:

1. Read B's `RELATED_TO`, `SHARED_ENTITY`, and `CAUSES` edges.
2. Replace endpoint B with A.
3. Preserve causal direction.
4. Normalize endpoint ordering for non-directional edges.
5. Skip self-edges created by replacement.
6. Upsert duplicate edges using the greater score.
7. Mark B inactive only after edge migration succeeds.

The old B edges may remain for auditing because B is inactive, but normal neighbor loading must filter inactive neighbors.

### 7.5 Contradiction resolution during reads

After anchors are selected and after every graph-expansion round:

1. Inspect lifecycle edges for `CONTRADICTS` relationships involving selected memories.
2. Load the active contradictory counterpart.
3. Add it once to evidence and visited IDs.
4. Let Jev's existing contradiction evidence assessment see both claims.

`CONTRADICTS` is not added to graph routing or graph budgets.

### 7.6 Jev response parsing

The current Jev parser supports only Noul answers. Consolidation requires Choice validation:

- selected option must be one of the requested criteria;
- every requested option must have a probability;
- every probability must be finite and in `[0, 1]`;
- probabilities must sum to approximately 1;
- selected option must be one of the highest-probability options;
- use the selected option probability for the merge threshold.

## 8. Background batch algorithm

```text
1. Create a consolidation run row.
2. Select up to batch_limit ACTIVE memories needing this decision_version.
3. Record them as run seeds.
4. For each seed:
   a. Find Top-K ACTIVE candidates.
   b. Canonicalize each pair as (min_id, max_id).
   c. Remove pairs already stored for this decision_version.
   d. Remove duplicate pairs proposed by another seed in this run.
   e. Batch remaining pair questions into a Jev request.
   f. Validate and store every decision before applying actions.
   g. Apply safe actions transactionally.
   h. Mark the seed processed for this version.
5. Complete the run and persist counts.
```

Candidate discovery for the first version should reuse the existing write-candidate logic: embedding similarity, entity overlap, and word overlap, bounded to Top-K. BM25 can be incorporated later if measurement shows it improves consolidation pair recall.

The run should select memories with `last_consolidated_version IS DISTINCT FROM current_version`, ordered with never-processed memories first. A new memory naturally compares itself with older active memories, so old seeds do not need to be reprocessed merely because one new memory arrived; the unordered pair is still discovered from the new side.

## 9. Transaction boundaries and invariants

Jev and generative model calls must not occur while holding long PostgreSQL transactions.

Recommended sequence:

```text
read candidate snapshot
  -> call Jev
  -> validate response
  -> start short transaction
  -> lock pair rows/memories
  -> recheck statuses and existing decision
  -> store decision and apply lifecycle mutation
  -> commit
```

For a generative merge:

```text
approved merge decision
  -> generate merged text outside transaction
  -> embed and prepare relation judgments outside transaction
  -> start short transaction
  -> recheck both sources are still active and pair is unresolved
  -> insert merged memory, lineage, edges, and source status changes atomically
  -> commit
```

Required invariants:

- every inactive memory has exactly one active replacement pointer;
- no memory replaces itself;
- replacement chains should be collapsed to one active target;
- contradiction never automatically deactivates either side;
- a merged result exists before its source memories become inactive;
- normal retrieval returns only active memories;
- pair decisions are unique per decision version;
- raw messages and inactive memories are never deleted by consolidation.

## 10. Read path after consolidation

```text
query
  -> Jev routes semantic/entity/causal graphs
  -> vector + BM25 anchor search over ACTIVE memories
  -> add active contradiction counterparts
  -> assess evidence
  -> if insufficient, expand knowledge edges to ACTIVE neighbors
  -> add contradiction counterparts for newly reached memories
  -> reassess
  -> format active resolved evidence
  -> answer
```

Redundant, merged-source, and superseded memories do not enter normal anchors or graph expansion. Their content remains accessible through explicit audit/provenance functions.

## 11. Implementation sequence used

Implement one concept at a time:

1. Add the schema migration and inspect it before applying.
2. Update loaders/searches to default to active memories and support nullable direct sources.
3. Add consolidation state/question construction, including Choice output.
4. Add strict Noul/Choice response parsing.
5. Add pair discovery, canonical pair keys, and decision persistence without applying actions.
6. Run one manual batch and inspect real Jev results.
7. Implement contradiction lifecycle edges and read-time counterpart inclusion.
8. Implement redundancy canonicalization and knowledge-edge migration.
9. Implement explicit supersession.
10. Implement optional generative merge and merged-memory lineage.
11. Wrap the manual batch function in a background/scheduled runner.
12. After core behavior works, add failure recovery, limits, metrics, and evaluation.

## 12. Manual scenarios to verify

### Redundancy

```text
The user's memory system uses PostgreSQL.
Postgres is the backend database for the user's memory system.
```

Expected: one active canonical memory, one inactive redundant memory, both source messages preserved.

### Merge

```text
The memory system uses PostgreSQL.
The memory system stores embeddings with pgvector.
```

Expected: one active merged memory containing both explicit details; both source memories inactive and linked to the result.

### Supersession

```text
The memory system uses SQLite.
The user migrated the memory system from SQLite to PostgreSQL.
```

Expected: PostgreSQL memory active; SQLite memory superseded and inactive.

### Contradiction

```text
The memory system uses PostgreSQL.
The memory system uses SQLite.
```

Expected: both active, one contradiction lifecycle edge, retrieving either includes both.

### Pair deduplication

Memory 1 proposes memory 3 as a candidate, then memory 3 proposes memory 1.

Expected: one `(1, 3, decision_version)` Jev decision.

### Concurrent worker safety

Two workers propose the same pair.

Expected: database uniqueness plus transactional recheck permits only one stored decision/action.

## 13. Decisions intentionally deferred

- promotion and behavioral abstraction;
- user-visible provenance citations;
- historical retrieval over superseded memories;
- automatic re-evaluation schedules for older decision versions;
- final thresholds beyond the initial 0.85 proposal;
- worker retry and dead-letter behavior;
- token and candidate budgets beyond bounded batch/Top-K inputs;
- whether inactive knowledge edges should eventually be compacted physically.
