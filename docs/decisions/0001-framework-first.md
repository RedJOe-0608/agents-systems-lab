# ADR 0001: Start framework-first behind local contracts

- **Status:** Accepted
- **Date:** 2026-10-02

## Context

The project aims to learn contemporary agent frameworks and ultimately implement selected
runtime components from first principles. Starting with a custom implementation would make it
difficult to distinguish novel design work from rediscovery of already-solved operational
problems.

## Decision

Build the first vertical slice with LangChain and LangGraph. Evaluate Deep Agents as an
opinionated harness baseline. Keep framework types behind local protocols for the runtime,
model gateway, context engine, sandbox, and tracing.

## Consequences

- A useful baseline can be delivered before the custom runtime exists.
- Framework behavior and replacement behavior can be compared on identical tasks.
- Adapters add some up-front structure.
- Dependency churn is contained at integration boundaries.
- A component will be replaced only with an explicit learning or performance objective.

