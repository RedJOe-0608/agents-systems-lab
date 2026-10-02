import json
import os

import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.environ["LLM_API_KEY"]
BASE_URL = os.environ["LLM_BASE_URL"]
MODEL_ID = os.environ["LLM_MODEL_ID"]


def count_r(text):
    """Count occurrences of the letter r, ignoring case."""
    return text.lower().count("r")


# This schema tells the model which tools are available.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "count_r",
            "description": (
                "Count occurrences of the letter r, ignoring case, "
                "in a string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact text to inspect.",
                    }
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    }
]


# This maps the model's function name to the actual Python function.
AVAILABLE_TOOLS = {
    "count_r": count_r,
}

MAX_STEPS = 5

# important observations:
# one model call can request multiple tool calls.
# Each tool call has a separate id


def call_model(messages):
    """Send the current conversation to the model."""

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_ID,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
        },
        timeout=120,
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]


def main():
    user_message = input("You: ")

    # Messages are the agent's current state.
    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    # Each iteration is one agent step.
    for step in range(MAX_STEPS):
        print(f"\n--- Agent step {step + 1} ---")

        assistant_message = call_model(messages)
        tool_calls = assistant_message.get("tool_calls")

        # No tool request means the model has produced its final answer.
        if not tool_calls:
            print(f"\nAssistant: {assistant_message['content']}")
            return

        # Preserve the model's tool request in the conversation.
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.get("content"),
                "tool_calls": tool_calls,
            }
        )

        print(f"Tool calls: {tool_calls}")

        # Execute all tools requested during this step.
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]

            arguments = json.loads(
                tool_call["function"]["arguments"]
            )

            # Basic policy check: only registered tools may execute.
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

            # Add the tool observation to the agent's state.
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": json.dumps(tool_result),
                }
            )

        # The loop now returns to call_model(messages).
        # The model sees the new observations and chooses what to do next.

    raise RuntimeError(
        f"Agent exceeded its limit of {MAX_STEPS} steps"
    )


if __name__ == "__main__":
    main()