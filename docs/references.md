# Source register

Last reviewed: **2026-10-02**.

Implementation decisions should be checked against current official documentation or primary
research when the corresponding experiment begins. Package APIs and public-beta features can
change; the lockfile records the versions actually evaluated.

## LangChain ecosystem

- [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Deep Agents overview](https://docs.langchain.com/oss/python/deepagents/overview)
- [LangSmith observability](https://docs.langchain.com/langsmith/observability)
- [LangSmith evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts)
- [LangChain learning paths and multi-agent patterns](https://docs.langchain.com/oss/python/learn)

## Primary research and implementations

- Wang et al., [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030)
- Zhang, Kraska, and Khattab,
  [Recursive Language Models](https://arxiv.org/abs/2512.24601)
- [Official RLM implementation](https://github.com/alexzhang13/rlm)
- [LangChain Deep Agents implementation](https://github.com/langchain-ai/deepagents)

## Documentation policy

Each experiment records:

- Date the documentation was checked
- Dependency versions from `uv.lock`
- Relevant official documentation or primary paper
- Any divergence from the documented default behavior

