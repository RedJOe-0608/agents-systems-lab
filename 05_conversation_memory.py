"""
Turn       = one user request through final answer
Step       = one LLM decision/model call
Tool call  = one requested function execution
Observation = one tool result
Message    = one item stored in conversation context
"""

import json
import os
import hashlib
import requests
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.environ["LLM_API_KEY"]
BASE_URL = os.environ["LLM_BASE_URL"]
MODEL_ID = os.environ["LLM_MODEL_ID"]


def count_r(text):
    """Count occurrences of the letter r, ignoring case."""
    return text.lower().count("r")

def calculate_sha256(text):
    """Calculate the SHA-256 hash of a string."""
    encoded_text = text.encode("utf-8")
    return hashlib.sha256(encoded_text).hexdigest()


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
    },
    {
    "type": "function",
        "function": {
            "name": "calculate_sha256",
            "description": (
                "Calculate the exact SHA-256 hash of a string "
                "and return its hexadecimal digest."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The exact text to hash.",
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
    "calculate_sha256": calculate_sha256,
}

MAX_STEPS = 5


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


def run_agent_turn(messages):
    """
    Run the agent loop for one user turn.

    The messages list is shared with the conversation loop,
    so tool calls, observations, and final answers remain in context.
    """

    # maximum number of times the model can be called to resolve one user message
    for step in range(MAX_STEPS):
        print(f"\n--- Agent step {step + 1} ---")

        assistant_message = call_model(messages)
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

            return final_content

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
        f"Agent exceeded its limit of {MAX_STEPS} steps"
    )


def main():
    # Conversation state starts empty.
    messages = []

    print("Agent started. Type 'quit' to stop.")

    while True:
        user_message = input("\nYou: ").strip()

        if user_message.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        if not user_message:
            continue

        # Add the new user message to the existing conversation.
        messages.append(
            {
                "role": "user",
                "content": user_message,
            }
        )

        final_answer = run_agent_turn(messages)

        print(f"\nAssistant: {final_answer}")
        print(f"[Messages currently in context: {len(messages)}]")


if __name__ == "__main__":
    main()