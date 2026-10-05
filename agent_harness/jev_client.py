import requests

from agent_harness.config import TYPESAFE_API_KEY


JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def evaluate_jev(state: dict, questions: dict) -> dict:
    response = requests.post(
        JEV_ENDPOINT,
        headers={"Authorization": f"Bearer {TYPESAFE_API_KEY}"},
        json={
            "model": "jev-latest",
            "state": state,
            "questions": questions,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()