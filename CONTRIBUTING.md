# Contributing

## Development workflow

```bash
uv sync --all-groups
uv run ruff check .
uv run mypy
uv run pytest
```

Use small commits that leave the repository runnable. Never commit credentials, generated
agent artifacts, model transcripts containing sensitive data, or an executable sandbox's
working directory.

## Experiment standard

Every experiment must document:

1. Question and hypothesis
2. Implementation and dependency versions
3. Dataset or task fixtures
4. Metrics and expected failure modes
5. Results and limitations
6. Decision: adopt, revise, or reject

Prefer deterministic evaluators for objective behavior. If an LLM judge is used, retain its
rubric, model identifier, sampling configuration, and raw score alongside the result.

