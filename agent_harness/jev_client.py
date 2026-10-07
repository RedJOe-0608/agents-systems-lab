import math
import requests

from agent_harness.config import TYPESAFE_API_KEY


JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
RELATION_THRESHOLD = 0.60

def build_relation_state(
    new_text: str,
    new_entities: list[dict],
    write_candidates: list[tuple],
) -> dict:
    candidates = []

    for row, _score in write_candidates:
        candidates.append({
            "memory_id": row[0],
            "content": row[1],
            "entities": row[2],
        })

    return {
        "new_memory": {
            "content": new_text,
            "entities": new_entities,
        },
        "candidates": candidates,
    }

def build_relation_questions(candidate_count: int) -> dict:
    questions = {}

    for index in range(candidate_count):
        pair = f"Compare `new_memory` with `candidates[{index}]`. "

        questions[f"pair_{index}_semantic"] = {
            "type": "noul",
            "instructions": (
                pair
                + "Would a semantic link help retrieve a specific shared topic or fact?"
            ),
            "criteria": {
                "true": (
                    "The memories share a specific topic, fact, goal, preference, "
                    "project, or event that makes the connection useful."
                ),
                "false": (
                    "They only share generic wording, or the connection would not "
                    "help future retrieval."
                ),
            },
        }

        questions[f"pair_{index}_causes"] = {
            "type": "noul",
            "instructions": (
                pair
                + "Does the new memory cause, enable, or explain the candidate memory?"
            ),
            "criteria": {
                "true": (
                    "The supplied memories support this direction of causation."
                ),
                "false": (
                    "There is only similarity, a shared entity, chronology, "
                    "or insufficient causal evidence."
                ),
            },
        }

        questions[f"pair_{index}_caused_by"] = {
            "type": "noul",
            "instructions": (
                pair
                + "Does the candidate memory cause, enable, or explain the new memory?"
            ),
            "criteria": {
                "true": (
                    "The supplied memories support this direction of causation."
                ),
                "false": (
                    "There is only similarity, a shared entity, chronology, "
                    "or insufficient causal evidence."
                ),
            },
        }

        questions[f"pair_{index}_shared_entity"] = {
            "type": "noul",
            "instructions": (
                pair
                + "Considering both the entity names and their contexts, do these "
                "memories refer to the same real-world entity, and would connecting "
                "them help future retrieval?"
            ),
            "criteria": {
                "true": (
                    "At least one entity refers to the same person, place, object, "
                    "organization, project, tool, or other referent in both memories, "
                    "and the connection would be useful."
                ),
                "false": (
                    "The entities refer to different things, their contexts conflict, "
                    "their identity is uncertain, or a matching name alone is not "
                    "enough to justify the connection."
                ),
            },
        }

    return questions

def evaluate_memory_relations(
    new_text: str,
    new_entities: list[dict],
    write_candidates: list[tuple],
) -> list[dict]:
    state = build_relation_state(
        new_text,
        new_entities,
        write_candidates,
    )

    candidate_count = len(state["candidates"])

    if candidate_count == 0:
        return []

    questions = build_relation_questions(candidate_count)
    response = evaluate_jev(state, questions)

    return parse_relation_answers(response, write_candidates)

def parse_relation_answers(
    response: dict,
    write_candidates: list[tuple],
) -> list[dict]:
    if not isinstance(response, dict):
        raise ValueError("Jev response must be an object")

    answers = response.get("answers")

    if not isinstance(answers, dict):
        raise ValueError("Jev response must contain an answers object")

    decisions = []

    for index, (row, _score) in enumerate(write_candidates):
        probabilities = {}

        for relation in (
            "semantic",
            "causes",
            "caused_by",
            "shared_entity",
        ):
            question_id = f"pair_{index}_{relation}"
            answer = answers.get(question_id)

            if not isinstance(answer, dict):
                raise ValueError(f"Missing Jev answer: {question_id}")

            if answer.get("type") != "noul":
                raise ValueError(
                    f"Jev answer must be a noul: {question_id}"
                )

            probability = answer.get("noul")

            if (
                type(probability) not in {int, float}
                or not math.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise ValueError(
                    f"Invalid Jev probability: {question_id}"
                )

            probabilities[relation] = float(probability)

        decisions.append({
            "existing_memory_id": row[0],
            **probabilities,
        })

    return decisions


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

def build_relation_edges(
    new_memory_id: int,
    decisions: list[dict],
    threshold: float = RELATION_THRESHOLD,
) -> list[dict]:
    edges = []

    for decision in decisions:
        existing_memory_id = decision["existing_memory_id"]

        # Non-directional edges must use a consistent ID order because
        # the database requires source_memory_id < target_memory_id.
        lower_id = min(new_memory_id, existing_memory_id)
        higher_id = max(new_memory_id, existing_memory_id)

        if decision["semantic"] >= threshold:
            edges.append({
                "source_memory_id": lower_id,
                "target_memory_id": higher_id,
                "relation_type": "RELATED_TO",
                "score": decision["semantic"],
            })

        if decision["shared_entity"] >= threshold:
            edges.append({
                "source_memory_id": lower_id,
                "target_memory_id": higher_id,
                "relation_type": "SHARED_ENTITY",
                "score": decision["shared_entity"],
            })

        if decision["causes"] >= threshold:
            edges.append({
                "source_memory_id": new_memory_id,
                "target_memory_id": existing_memory_id,
                "relation_type": "CAUSES",
                "score": decision["causes"],
            })

        if decision["caused_by"] >= threshold:
            edges.append({
                "source_memory_id": existing_memory_id,
                "target_memory_id": new_memory_id,
                "relation_type": "CAUSES",
                "score": decision["caused_by"],
            })

    return edges