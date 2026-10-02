# Architecture overview

## Target workflow

```text
User/API
   |
Coordinator
   |-- repository analyst -- recursive context exploration
   |-- implementation agent -- sandboxed executable actions
   |-- test agent ---------- isolated verification
   `-- review agent -------- independent critique
                  |
          artifacts and event log
                  |
      evaluation and observability
```

The first implementation will use LangGraph for orchestration and LangChain model/tool
interfaces. Deep Agents will provide a second, more opinionated baseline. Both must adapt to
the interfaces in `agent_systems_lab.core`; domain code should not accept framework state
objects directly.

## Architectural boundaries

- **Agent runtime:** task lifecycle, coordination, retries, cancellation, and results.
- **Model gateway:** provider-independent model calls and tool schemas.
- **Context engine:** selection, decomposition, recursive model calls, and synthesis.
- **Sandbox:** capability-limited executable actions with resource budgets.
- **State and artifacts:** durable checkpoints and outputs too large for model context.
- **Trace sink:** framework-neutral events for replay, diagnostics, and evaluation.
- **Evaluation:** task fixtures and deterministic or explicitly versioned judges.

## Safety boundaries

Generated code is untrusted. A production CodeAct-style executor must not inherit host
credentials, unrestricted networking, the developer filesystem, or unlimited compute. Human
approval belongs at capability boundaries, not only in prompts.

## Replacement strategy

1. Measure the framework baseline.
2. Capture behavior with contract tests and event traces.
3. Replace one adapter at a time.
4. Run the same evaluation suite after each replacement.
5. Retain a framework implementation as a comparison until the custom path is superior or
   provides a clearly documented systems-learning benefit.

