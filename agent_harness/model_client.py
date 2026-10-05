import requests
import time

RETRYABLE_STATUS_CODES = {408, 409, 429}
# 408: server timed out processing the request
# 409: temporary conflict.
# 429: rate limit reached. 
# 5xx: any server-side failure

MAX_RETRIES = 2
INITIAL_RETRY_DELAY_SECONDS = 1

# Exceptions are classes. To behave like an exception, our class must inherit from an existing class.
class ModelRequestError(RuntimeError):
    """Raised when a model request cannot be completed."""

def _is_retryable_status(status_code):
    return (
        status_code in RETRYABLE_STATUS_CODES
        or 500 <= status_code < 600
    )

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

    for attempt in range(MAX_RETRIES + 1):
        # Retry when the client does not receive a usable HTTP response.
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=120,
            )
        # in python, it is try/except rather than try/catch. the way to read this is, if requests.Timeout exception occurs, or requests.ConnectionError occurs, catch it and store the object in error variable.
        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as error:
            retries_exhausted = attempt == MAX_RETRIES

            if retries_exhausted:
                raise ModelRequestError(
                "The model service could not be reached after "
                f"{MAX_RETRIES + 1} attempts."
            ) from error

            retry_delay = (
                INITIAL_RETRY_DELAY_SECONDS
                * (2 ** attempt)
            )

            print(
                f"Model request failed with "
                f"{type(error).__name__}. "
                f"Retrying in {retry_delay} seconds."
            )

            time.sleep(retry_delay)
            continue

        except requests.RequestException as error:
            raise ModelRequestError(
                "The model request failed before a response "
                "could be processed."
        ) from error

        should_retry = (
            _is_retryable_status(response.status_code)
            and attempt < MAX_RETRIES
        )

        if should_retry:
            retry_delay = (
                INITIAL_RETRY_DELAY_SECONDS
                * (2 ** attempt)
            )

            print(
                f"Model request failed with HTTP "
                f"{response.status_code}. "
                f"Retrying in {retry_delay} seconds."
            )

            time.sleep(retry_delay)
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as error:
            raise ModelRequestError(
                f"The model service returned HTTP "
                f"{response.status_code}."
            ) from error

        # Suppose the server replies with HTTP 200, but its response body is: {"choices": [
        # That text is incomplete JSON. response.raise_for_status() succeeds because HTTP 200 reports success, but response.json() cannot parse the body and raises requests.exceptions.JSONDecodeError.
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as error:
            raise ModelRequestError(
                "The model service returned invalid JSON."
            ) from error