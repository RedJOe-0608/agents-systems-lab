import requests

# * means the remaining parameters must be named, while invoking this function.
# handles HTTP request to the model and returns complete response data.
def request_chat_completion(
    *,
    base_url,
    api_key,
    model_id,
    messages,
    tools=None,
):
    """Send one request to a Chat Completions endpoint."""

    payload = {
        "model": model_id,
        "messages": messages,
    }

    # Normal agent calls will pass TOOLS, but summary calls will omit TOOLS.
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=120,
    )

    response.raise_for_status()

    return response.json()