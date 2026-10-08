# Agents Systems Lab

This repository is a learning project for building an agent harness and a
PostgreSQL-backed long-term memory system.

The current memory system supports extraction, embeddings, hybrid retrieval,
Jev-guided graph construction and traversal, and bounded background memory
consolidation. Consolidation can canonicalize redundant memories, preserve
contradictions, apply explicit supersession, and create source-linked merged
memories without deleting raw evidence.

Run one consolidation batch manually:

```bash
.venv/bin/python run_memory_consolidation.py \
  --batch-limit 20 \
  --candidate-limit 10
```

The command prints a JSON report. `remaining_seed_count` tells you whether
another batch is needed. Add `--decide-only` to persist Jev decisions without
applying lifecycle actions; a later normal run will apply pending decisions.

Consolidation calls the configured Jev and language-model services and mutates
memory lifecycle state. It is never invoked automatically by `run_agent.py`.

Run the core policy and PostgreSQL transaction tests with:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The implementation map and design rationale are in
`docs/plans/memory-consolidation.md`.
