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

    messages = [
        {
            "role": "user",
            "content": user_message,
        }
    ]

    # First model call: the model may answer directly or request a tool.
    assistant_message = call_model(messages)

    tool_calls = assistant_message.get("tool_calls")

    # The model decided that no tool was necessary.
    if not tool_calls:
        print(f"\nAssistant: {assistant_message['content']}")
        return

    # Preserve the assistant's tool request in the conversation.
    messages.append(
        {
            "role": "assistant",
            "content": assistant_message.get("content"),
            "tool_calls": tool_calls,
        }
    )

    # Execute every tool requested by the model.
    for tool_call in tool_calls:
        function_name = tool_call["function"]["name"]

        # The arguments arrive as a JSON string.
        arguments = json.loads(
            tool_call["function"]["arguments"]
        )

        if function_name not in AVAILABLE_TOOLS:
            raise ValueError(f"Unknown tool: {function_name}")

        python_function = AVAILABLE_TOOLS[function_name]
        tool_result = python_function(**arguments)

        print(
            f"\nPython executed: "
            f"{function_name}({arguments})"
        )
        print(f"Tool result: {tool_result}")

        # Return the result to the model.
        # tool_call_id connects this result to the original request.
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "name": function_name,
                "content": json.dumps(tool_result),
            }
        )

    # Second model call: the model can now use the tool result.
    final_message = call_model(messages)

    print(f"\nAssistant: {final_message['content']}")


if __name__ == "__main__":
    main()