# Agents Systems Lab

A framework-first laboratory for learning, evaluating, and eventually implementing a
multi-agent runtime from first principles.

The project starts with LangChain, LangGraph, Deep Agents, and LangSmith to establish a
working baseline. It then studies CodeAct-style executable actions and Recursive Language
Models (RLMs), measures where each technique helps, and replaces selected framework
components behind stable local interfaces.

## Why this repository exists

This is not a collection of disconnected agent demos. The goal is to build one coherent
software-engineering agent while preserving the experiments and evidence that inform its
architecture.

The system will ultimately accept a repository task, investigate the codebase, propose and
implement a change in an isolated environment, run verification, and produce a reviewable
result.

## Principles

- Establish a measured framework baseline before writing a custom runtime.
- Use multiple agents only when isolation or specialization improves results.
- Treat executable code as untrusted and run it behind capability boundaries.
- Evaluate quality, latency, token use, cost, and recovery behavior.
- Keep model, orchestration, context, sandbox, and tracing interfaces replaceable.
- Record important decisions and cite primary or official sources.

## Repository map

```text
src/agent_systems_lab/
  core/                 Framework-independent contracts and domain models
experiments/            Incremental, reproducible learning experiments
evaluations/            Datasets, metrics, and comparison harnesses
docs/architecture/      Current system design
docs/decisions/         Architecture decision records
tests/                  Fast deterministic tests
```

## Quick start

Prerequisites: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups
uv run agents-lab doctor
uv run pytest
uv run ruff check .
uv run mypy
```

No API key is required for the initial checks. When an experiment needs a model, copy
`.env.example` to `.env` and add credentials locally. The `.env` file is ignored by Git.

## Learning sequence

1. A minimal LangChain tool-calling agent.
2. The same workflow expressed explicitly with LangGraph.
3. A Deep Agents baseline for long-horizon work.
4. A sandboxed CodeAct-style execution loop.
5. An RLM-inspired recursive context engine.
6. Single-agent and multi-agent comparative evaluation.
7. Incremental replacement with a custom scheduler, state store, and trace model.

See [the learning roadmap](docs/learning-roadmap.md), [the architecture](docs/architecture/overview.md),
and [the source register](docs/references.md).

## Current status

**Phase 0 — foundation.** Package boundaries, development tooling, documentation, and
framework-independent contracts are in place. The first functional experiment is next.

## License

[MIT](LICENSE)

