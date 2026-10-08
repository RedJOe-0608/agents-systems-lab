import math
import requests

from agent_harness.config import TYPESAFE_API_KEY


JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
RELATION_THRESHOLD = 0.60
GRAPH_ACTIVATION_THRESHOLD = 0.10
MAXIMUM_GRAPH_DEPTH = 8
TOTAL_GRAPH_BUDGET = 80
MINIMUM_GRAPH_BUDGET = 1
EVIDENCE_SUFFICIENT_THRESHOLD = 0.95
EVIDENCE_ISSUE_THRESHOLD = 0.15

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

def build_query_routing_questions() -> dict:
    return {
        "semantic": {
            "type": "noul",
            "instructions": (
                "Would following semantically related memories help answer `query`?"
            ),
            "criteria": {
                "true": (
                    "The query requires facts, preferences, goals, projects, or "
                    "topics that may be connected through semantic relationships."
                ),
                "false": (
                    "Semantically related memories would not provide useful "
                    "evidence for answering the query."
                ),
            },
        },

        "causal": {
            "type": "noul",
            "instructions": (
                "Does answering `query` require finding a cause, reason, "
                "explanation, enabling condition, or consequence?"
            ),
            "criteria": {
                "true": (
                    "The query requires causal or explanatory evidence."
                ),
                "false": (
                    "The query can be answered without following causal relationships."
                ),
            },
        },

        "entity": {
            "type": "noul",
            "instructions": (
                "Would connecting memories about the same person, place, project, "
                "tool, organization, object, or other entity help answer `query`?"
            ),
            "criteria": {
                "true": (
                    "Combining facts associated with the same entity would help "
                    "answer the query."
                ),
                "false": (
                    "Entity identity and entity-linked memories are not useful "
                    "for answering the query."
                ),
            },
        },

        "multi_hop_need": {
            "type": "noul",
            "instructions": (
                "Does answering `query` require combining two or more distinct "
                "pieces of remembered evidence?"
            ),
            "criteria": {
                "true": (
                    "The answer requires combining, comparing, aggregating, or "
                    "chaining multiple memories."
                ),
                "false": (
                    "One directly relevant memory should be sufficient."
                ),
            },
        },
    }

def evaluate_query_routing(query: str) -> dict:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query must be non-empty text")

    state = {
        "query": query.strip(),
    }

    questions = build_query_routing_questions()
    response = evaluate_jev(state, questions)

    return parse_query_routing_answers(response)

def parse_query_routing_answers(response: dict) -> dict:
    if not isinstance(response, dict):
        raise ValueError("Jev routing response must be an object")

    answers = response.get("answers")

    if not isinstance(answers, dict):
        raise ValueError(
            "Jev routing response must contain an answers object"
        )

    routing = {}

    for graph_need in (
        "semantic",
        "causal",
        "entity",
        "multi_hop_need",
    ):
        answer = answers.get(graph_need)

        if not isinstance(answer, dict):
            raise ValueError(
                f"Missing Jev routing answer: {graph_need}"
            )

        if answer.get("type") != "noul":
            raise ValueError(
                f"Jev routing answer must be a noul: {graph_need}"
            )

        probability = answer.get("noul")

        if (
            type(probability) not in {int, float}
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError(
                f"Invalid Jev routing probability: {graph_need}"
            )

        routing[graph_need] = float(probability)

    return routing

def activate_query_graphs(
    routing: dict,
    threshold: float = GRAPH_ACTIVATION_THRESHOLD,
) -> dict:
    active_graphs = {}

    for graph in (
        "semantic",
        "causal",
        "entity",
    ):
        probability = routing[graph]

        if probability >= threshold:
            active_graphs[graph] = probability

    return active_graphs

def calculate_query_depth_limit(
    routing: dict,
    maximum_depth: int = MAXIMUM_GRAPH_DEPTH,
) -> int:
    multi_hop_need = routing["multi_hop_need"]

    return min(
        maximum_depth,
        max(
            1,
            math.ceil(maximum_depth * multi_hop_need),
        ),
    )

def allocate_query_graph_budgets(
    active_graphs: dict,
    total_budget: int = TOTAL_GRAPH_BUDGET,
    minimum_budget: int = MINIMUM_GRAPH_BUDGET,
) -> dict:
    budgets = {
        "semantic": 0,
        "causal": 0,
        "entity": 0,
    }

    if not active_graphs or total_budget == 0:
        return budgets

    graph_names = sorted(
        active_graphs,
        key=lambda graph: (
            -active_graphs[graph],
            graph,
        ),
    )

    # Ensure we do not activate more graphs than the total budget can support.
    if minimum_budget > 0:
        maximum_active_graphs = max(
            1,
            total_budget // minimum_budget,
        )
        graph_names = graph_names[:maximum_active_graphs]
        minimum_budget = min(minimum_budget, total_budget)

    for graph in graph_names:
        budgets[graph] = minimum_budget

    remaining_budget = total_budget - sum(budgets.values())

    if remaining_budget == 0:
        return budgets

    total_probability = sum(
        active_graphs[graph]
        for graph in graph_names
    )

    shares = {
        graph: (
            remaining_budget
            * active_graphs[graph]
            / total_probability
        )
        for graph in graph_names
    }

    for graph in graph_names:
        budgets[graph] += int(shares[graph])

    leftover_budget = total_budget - sum(budgets.values())

    remainder_order = sorted(
        graph_names,
        key=lambda graph: (
            -(shares[graph] % 1),
            graph,
        ),
    )

    for graph in remainder_order[:leftover_budget]:
        budgets[graph] += 1

    return budgets

def build_evidence_assessment_questions() -> dict:
    return {
        "evidence_sufficient": {
            "type": "noul",
            "instructions": (
                "Does `evidence` contain support for every factual part "
                "needed to answer `query`?"
            ),
            "criteria": {
                "true": (
                    "The query can be answered from the supplied evidence "
                    "without inventing missing facts."
                ),
                "false": (
                    "At least one fact needed for the answer is unsupported."
                ),
            },
        },

        "missing_evidence": {
            "type": "noul",
            "instructions": (
                "Is at least one fact required to answer `query` absent "
                "from `evidence`?"
            ),
            "criteria": {
                "true": (
                    "A required fact, identity, reason, detail, or connecting "
                    "piece of information is missing."
                ),
                "false": (
                    "Every fact needed for the answer is explicitly supported "
                    "by the supplied evidence."
                ),
            },
        },

        "contradiction": {
            "type": "noul",
            "instructions": (
                "Does `evidence` contain conflicting claims relevant to `query`?"
            ),
            "criteria": {
                "true": (
                    "The evidence contains incompatible claims that affect "
                    "which answer is correct."
                ),
                "false": (
                    "The evidence is compatible, or any differences do not "
                    "affect the answer."
                ),
            },
        },
    }

def build_evidence_assessment_state(
    query: str,
    evidence_rows: list[tuple],
    depth: int,
) -> dict:
    evidence = []

    for row in evidence_rows:
        evidence.append({
            "memory_id": row[0],
            "content": row[1],
            "entities": row[2],
        })

    return {
        "query": query,
        "evidence": evidence,
        "depth": depth,
    }

def evaluate_evidence_sufficiency(
    query: str,
    evidence_rows: list[tuple],
    depth: int,
) -> dict:
    if not evidence_rows:
        raise ValueError(
            "Evidence assessment requires at least one memory"
        )

    state = build_evidence_assessment_state(
        query,
        evidence_rows,
        depth,
    )

    questions = build_evidence_assessment_questions()
    response = evaluate_jev(state, questions)

    return parse_evidence_assessment_answers(response)

def parse_evidence_assessment_answers(
    response: dict,
) -> dict:
    if not isinstance(response, dict):
        raise ValueError(
            "Jev evidence response must be an object"
        )

    answers = response.get("answers")

    if not isinstance(answers, dict):
        raise ValueError(
            "Jev evidence response must contain an answers object"
        )

    assessment = {}

    for field in (
        "evidence_sufficient",
        "missing_evidence",
        "contradiction",
    ):
        answer = answers.get(field)

        if not isinstance(answer, dict):
            raise ValueError(
                f"Missing Jev evidence answer: {field}"
            )

        if answer.get("type") != "noul":
            raise ValueError(
                f"Jev evidence answer must be a noul: {field}"
            )

        probability = answer.get("noul")

        if (
            type(probability) not in {int, float}
            or not math.isfinite(probability)
            or not 0.0 <= probability <= 1.0
        ):
            raise ValueError(
                f"Invalid Jev evidence probability: {field}"
            )

        assessment[field] = float(probability)

    return assessment

def is_evidence_sufficient(
    assessment: dict,
    sufficient_threshold: float = EVIDENCE_SUFFICIENT_THRESHOLD,
    issue_threshold: float = EVIDENCE_ISSUE_THRESHOLD,
) -> bool:
    return (
        assessment["evidence_sufficient"] >= sufficient_threshold
        and assessment["missing_evidence"] < issue_threshold
        and assessment["contradiction"] < issue_threshold
    )