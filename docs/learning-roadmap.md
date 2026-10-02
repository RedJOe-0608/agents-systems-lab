# Learning and delivery roadmap

## Phase 0: Foundation

- Reproducible Python environment and CI
- Framework-independent contracts
- Architecture decision records and source register
- Deterministic unit tests

## Phase 1: Tool-calling baseline

- Implement a minimal LangChain agent
- Learn messages, tool schemas, middleware, structured output, and model adapters
- Record complete traces and create the first evaluation fixture

## Phase 2: Explicit orchestration

- Rebuild the workflow with LangGraph
- Study state, reducers, nodes, edges, interrupts, persistence, and streaming
- Compare behavior and complexity with the Phase 1 agent

## Phase 3: Agent harness

- Implement the same task using Deep Agents
- Study filesystem backends, context compaction, subagents, permissions, and memory
- Identify which harness defaults materially improve the benchmark

## Phase 4: Executable actions

- Build a CodeAct-style loop around an isolated executor
- Add CPU, memory, time, filesystem, network, and credential boundaries
- Compare composed code actions against one-tool-call-at-a-time behavior

## Phase 5: Recursive context

- Treat a repository snapshot as external context
- Let a root model inspect, partition, and invoke bounded subcalls over selected content
- Measure accuracy, latency, recursion depth, and context/token consumption

## Phase 6: Multi-agent evaluation

- Add analyst, implementer, test, and reviewer roles only where isolation is useful
- Compare single-agent, sequential multi-agent, and parallel variants
- Test recovery from tool failure, invalid patches, timeouts, and context pressure

## Phase 7: Custom runtime

- Replace orchestration, state, tracing, and context components incrementally
- Preserve trace replay and evaluation compatibility
- Document measured tradeoffs against the framework baselines

