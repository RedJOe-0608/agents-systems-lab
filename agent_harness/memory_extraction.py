import json
from agent_harness.config import API_KEY, BASE_URL, MODEL_ID
from agent_harness.db import load_session_messages
from agent_harness.model_client import request_chat_completion

def parse_extraction(raw_response: str, session_rows) -> list[dict]:
    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Extractor returned invalid JSON") from error

    if not isinstance(data, dict) or not isinstance(data.get("memories"), list):
        raise ValueError("Extractor response must contain a memories list")

    allowed_ids = {
        message_id
        for message_id, _, role, payload in session_rows
        if role in {"user", "assistant"}
        and isinstance(payload.get("content"), str)
        and payload["content"].strip()
    }

    candidates = []
    for item in data["memories"]:
        if not isinstance(item, dict):
            raise ValueError("Each memory must be an object")

        text = item.get("text")
        source_id = item.get("source_message_id")

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Memory text must be non-empty")
        if type(source_id) is not int or source_id not in allowed_ids:
            raise ValueError("Memory source must be a supplied message ID")

        candidates.append({"text": text.strip(), "source_message_id": source_id})

    return candidates

EXTRACTION_SYSTEM_PROMPT = """
Extract durable facts that could help in future conversations.

Include user goals, preferences, ongoing projects, and decisions the user made.
Write each fact as one self-contained sentence.
For each fact, give the ID of a message that directly supports it.
Use only message IDs supplied in the conversation.
Skip guesses, temporary requests, and secrets.

Return only JSON in this shape:
{"memories": [{"text": "The user is building a long-term memory project.", "source_message_id": 123}]}

If there are no useful facts, return {"memories": []}.
""".strip()

def build_extraction_input(session_rows) -> str:
    messages = []

    for message_id, sequence_no, role, payload in session_rows:
        content = payload.get("content")

        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({
                "message_id": message_id,
                "role": role,
                "content": content,
            })

    return json.dumps({"messages": messages}, ensure_ascii=False)

def request_extraction(session_rows) -> str:
    response = request_chat_completion(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_id=MODEL_ID,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": build_extraction_input(session_rows)},
        ],
    )
    return response["choices"][0]["message"]["content"]

def extract_session_candidates(session_id: int) -> list[dict]:
    session_rows = load_session_messages(session_id)
    raw_response = request_extraction(session_rows)
    return parse_extraction(raw_response, session_rows)