import json
from agent_harness.config import API_KEY, BASE_URL, MODEL_ID
from agent_harness.model_client import request_chat_completion
from agent_harness.storage.conversations import (
    load_session_messages,
)

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

        entities = item.get("entities")
        if not isinstance(entities, list):
            raise ValueError("Memory entities must be a list")

        clean_entities = []
        for entity in entities:
            if not isinstance(entity, dict):
                raise ValueError("Each entity must be an object")

            name = entity.get("name")
            context = entity.get("context")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Entity name must be non-empty")
            if not isinstance(context, str) or not context.strip():
                raise ValueError("Entity context must be non-empty")

            clean_entities.append({
                "name": name.strip(),
                "context": context.strip(),
            })

        candidates.append({
            "text": text.strip(),
            "source_message_id": source_id,
            "entities": clean_entities,
        })

    return candidates

EXTRACTION_SYSTEM_PROMPT = """
Extract durable facts that could help in future conversations.

Include user goals, preferences, ongoing projects, and decisions the user made.
Write each fact as one self-contained sentence.
For each fact, give the ID of one message that directly supports it.
Use only message IDs supplied in the conversation.
Skip guesses, temporary requests, and secrets.

For each memory, list entities central to that fact and explicitly mentioned
in its source message. An entity may be a person, animal, food, place,
organization, project, tool, or other referent. Do not list every incidental
noun. Use the name as it appears in the source message and add a short context
explaining what it refers to. Do not invent entities or resolve aliases.
Use an empty entities list when none qualify.

Return only valid JSON, with no Markdown, in this shape:
{
  "memories": [
    {
      "text": "The user is building a memory system inspired by Jev-Mem.",
      "source_message_id": 123,
      "entities": [
        {
          "name": "Jev-Mem",
          "context": "The memory system inspiring the user's project."
        }
      ]
    }
  ]
}

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
