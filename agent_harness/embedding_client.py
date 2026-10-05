import requests

from agent_harness.config import API_KEY, BASE_URL, EMBEDDING_MODEL_ID
from agent_harness.model_client import ModelRequestError


def embed_text(text: str) -> list[float]:
    try:
        response = requests.post(
            f"{BASE_URL.rstrip('/')}/v1/embeddings",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model": EMBEDDING_MODEL_ID, "input": text},
            timeout=60,
        )
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as error:
        raise ModelRequestError("Embedding request failed") from error

    if not isinstance(embedding, list) or len(embedding) != 2048:
        raise ModelRequestError("Embedding response must contain 2048 values")

    return embedding