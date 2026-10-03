import json
import time

from agent_harness.config import API_KEY, BASE_URL, MODEL_ID
from agent_harness.context import build_context
from agent_harness.model_client import request_chat_completion
from agent_harness.tools import AVAILABLE_TOOLS, TOOLS


def call_model(messages):
    """Call the model with the agent's available tools."""

    data = request_chat_completion(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_id=MODEL_ID,
        messages=messages,
        tools=TOOLS,
    )

    assistant_message = data["choices"][0]["message"]
    usage = data.get("usage", {})

    return assistant_message, usage

def run_agent_turn(messages, conversation_summary, summarized_until, *, max_steps):
    """
    Run the agent loop for one user turn.

    The messages list is shared with the conversation loop,
    so tool calls, observations, and final answers remain in context.
    """

    turn_usage = {
        "model_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "model_seconds": 0.0,
    }

    # maximum number of times the model can be called to resolve one user message
    for step in range(max_steps):
        print(f"\n--- Agent step {step + 1} ---")

        context_messages = build_context(messages, conversation_summary, summarized_until)

        print(f"Messages stored: {len(messages)}")
        print(f"Context messages sent: {len(context_messages)}")

        start_time = time.perf_counter()
        # We only send the context messages to the model, not the full messages list. So the last MAX_CONTEXT_TURNS are sent to the model.
        assistant_message, usage = call_model(context_messages)
        elapsed_seconds = time.perf_counter() - start_time

        turn_usage["model_calls"] += 1
        turn_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        turn_usage["completion_tokens"] += usage.get(
            "completion_tokens", 0
        )
        turn_usage["total_tokens"] += usage.get("total_tokens", 0)
        turn_usage["model_seconds"] += elapsed_seconds

        tool_calls = assistant_message.get("tool_calls")

        # No tool call means the model has finished this turn.
        if not tool_calls:
            final_content = assistant_message["content"]

            # Store the final response so future user messages can refer to it.
            messages.append(
                {
                    "role": "assistant",
                    "content": final_content,
                }
            )

            return final_content, turn_usage

        # Store the model's tool requests.
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.get("content"),
                "tool_calls": tool_calls,
            }
        )

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]

            arguments = json.loads(
                tool_call["function"]["arguments"]
            )

            if function_name not in AVAILABLE_TOOLS:
                raise ValueError(
                    f"Tool is not allowed: {function_name}"
                )

            python_function = AVAILABLE_TOOLS[function_name]
            tool_result = python_function(**arguments)

            print(
                f"Executing: {function_name}({arguments})"
            )
            print(f"Observation: {tool_result}")

            # Store the observation so the model can use it.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": json.dumps(tool_result),
                }
            )

    raise RuntimeError(
        f"Agent exceeded its limit of {max_steps} steps"
    )
