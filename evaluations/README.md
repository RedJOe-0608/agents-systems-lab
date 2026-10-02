# Evaluations

The evaluation suite will compare architecture variants on identical, reproducible repository
tasks.

Initial metrics:

- Objective task success and test pass rate
- Patch correctness and regression count
- Human interventions
- Model and tool calls
- Input/output tokens and estimated cost
- Wall-clock latency
- Retry and recovery rate
- Context compression or retrieval failures

Deterministic checks take precedence over LLM judges. Subjective judges must have a versioned
rubric and may not be the sole measure of task success.

