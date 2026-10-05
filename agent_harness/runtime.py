import json
import time
import inspect

from agent_harness.config import API_KEY, BASE_URL, MODEL_ID
from agent_harness.context import build_context
from agent_harness.model_client import request_chat_completion
from agent_harness.tools import AVAILABLE_TOOLS, TOOLS

class RepeatedToolCallError(RuntimeError):
    """Raised when the model repeats an identical tool request."""

MAX_IDENTICAL_TOOL_REQUESTS = 3

class AgentStepLimitError(RuntimeError):
    """Raised when a turn cannot finish within its step limit."""

def _append_tool_error(
    messages,
    tool_call,
    function_name,
    error_type,
    error_message,
):
    """Return a structured tool error to the model."""

    error_observation = {
        "ok": False,
        "error": {
            "type": error_type,
            "message": error_message,
        },
    }

    print(
        f"Tool error: {function_name} — "
        f"{error_message}"
    )

    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "name": function_name,
            "content": json.dumps(error_observation),
        }
    )


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

    previous_tool_request = None
    repeat_count = 0

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

        current_tool_request = tuple(
            (
                tool_call["function"]["name"],
                tool_call["function"].get("arguments"),
            )
            for tool_call in tool_calls
        )

        if current_tool_request == previous_tool_request:
            repeat_count += 1
        else:
            previous_tool_request = current_tool_request
            repeat_count = 1

        if repeat_count >= MAX_IDENTICAL_TOOL_REQUESTS:
            raise RepeatedToolCallError(
                "The model repeated the same tool request "
                f"{repeat_count} times."
            )

        if step == max_steps - 1:
            raise AgentStepLimitError(
                f"Agent reached its limit of {max_steps} steps "
                "without a final answer."
            )

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

            raw_arguments = tool_call["function"].get("arguments")

           # Here, we are checking, if the arguments that the LLM proposed are valid JSON.  
            try:
                arguments = json.loads(raw_arguments)
            except (json.JSONDecodeError, TypeError):
                _append_tool_error(
                    messages,
                    tool_call,
                    function_name,
                    "invalid_arguments",
                    (
                        "Tool arguments must be valid JSON. "
                        "Call the tool again with a JSON object "
                        "matching its schema."
                    ),
                )
                continue

            # here, we are checking, if the arguments, now that they are valid json, are they valid objects?
            if not isinstance(arguments, dict):
                _append_tool_error(
                    messages,
                    tool_call,
                    function_name,
                    "invalid_arguments",
                    (
                        "Tool arguments must be a JSON object. "
                        "Call the tool again with named arguments "
                        "matching its schema."
                    ),
                )
                continue

            # here, we check, if the model proposes a tool request that is not in the allowed list of tools, we reject it, rather than crashing our conversation
            if function_name not in AVAILABLE_TOOLS:
                _append_tool_error(
                    messages,
                    tool_call,
                    function_name,
                    "tool_not_allowed",
                    (
                        f"The tool '{function_name}' is not available. "
                        "Choose one of the provided tools or answer "
                        "without using a tool."
                    ),
                )
                continue

            python_function = AVAILABLE_TOOLS[function_name]

            # here, we check if the function argument names are correct.
            try:
                inspect.signature(python_function).bind(**arguments)
            except TypeError:
                _append_tool_error(
                    messages,
                    tool_call,
                    function_name,
                    "invalid_arguments",
                    (
                        "The provided arguments do not match the "
                        "tool's required named parameters. "
                        "Call the tool again using its schema."
                    ),
                )
                continue

            # this checks if the argument values are of the correct value.
            try:
                tool_result = python_function(**arguments)
            except Exception as error:
                error_name = type(error).__name__

                _append_tool_error(
                    messages,
                    tool_call,
                    function_name,
                    "tool_execution_error",
                    (
                        f"The tool failed during execution with "
                        f"{error_name}. Check the argument values, "
                        "then retry with corrected input or explain "
                        "that the tool could not complete."
                    ),
                )
                continue

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

    raise AgentStepLimitError(
        f"Agent exceeded its limit of {max_steps} steps"
    )
