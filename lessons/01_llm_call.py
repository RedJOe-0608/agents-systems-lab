import os

import requests
from dotenv import load_dotenv


# Load configuration values from the local .env file.
# The API key stays outside the Python source and outside Git.
load_dotenv()

API_KEY = os.environ["LLM_API_KEY"]
BASE_URL = os.environ["LLM_BASE_URL"]
MODEL_ID = os.environ["LLM_MODEL_ID"]


def main():
    # Collect one message from the user.
    user_message = input("You: ")

    # This is the JSON body sent to the LLM server.
    # At this point, the conversation contains only one user message.
    payload = {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": user_message,
            }
        ],
    }

    # The API key proves that we are authorized to use the endpoint.
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    # Send the request to the OpenAI-compatible chat-completions endpoint.
    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )

    # Raise an exception if the server returned an error response.
    response.raise_for_status()

    # Convert the JSON response into Python dictionaries and lists.
    data = response.json()

    # Extract only the assistant's final answer.
    assistant_message = data["choices"][0]["message"]["content"]

    print(f"\nAssistant: {assistant_message}")


# Run main() only when this file is executed directly.
if __name__ == "__main__":
    main()