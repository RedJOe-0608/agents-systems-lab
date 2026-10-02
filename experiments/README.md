# Experiments

Experiments progress from framework primitives to a custom runtime. Each directory will contain
a runnable entry point, task fixtures, a result record, and a short explanation of what was
learned.

| Experiment | Question |
| --- | --- |
| `01_basic_tool_agent` | What does LangChain's minimal agent loop provide? |
| `02_langgraph_workflow` | What becomes explicit when the loop is modeled as a graph? |
| `03_deep_agent` | Which harness capabilities improve long-horizon work? |
| `04_codeact_sandbox` | When are executable actions better than individual tool calls? |
| `05_recursive_context` | Can recursive context exploration outperform flat retrieval? |
| `06_multi_agent_comparison` | When does delegation outperform one capable agent? |

Do not compare experiments on ad hoc prompts. Reuse versioned fixtures from `evaluations/`.

