import json

from agent_harness.config import API_KEY, BASE_URL, MODEL_ID
from agent_harness.embedding_client import embed_text
from agent_harness.jev_client import (
    evaluate_consolidation_pairs,
    evaluate_memory_relations,
)
from agent_harness.memory_lifecycle import (
    apply_merge_decision,
    apply_non_merge_decision,
    choose_consolidation_action,
)
from agent_harness.model_client import request_chat_completion
from agent_harness.storage.consolidation import (
    count_pending_consolidation_seeds,
    create_consolidation_run,
    fail_consolidation_run,
    find_consolidation_candidates,
    finish_consolidation_run,
    load_consolidation_decision,
    load_consolidation_seeds,
    load_evaluated_pair_keys,
    load_pair_memory_snapshots,
    load_pending_consolidation_decisions,
    mark_consolidation_seeds_processed,
    record_consolidation_decision_error,
    record_consolidation_run_seeds,
    save_consolidation_decisions,
)
from agent_harness.storage.memories import find_write_candidates


CONSOLIDATION_DECISION_VERSION = "v1"
CONSOLIDATION_MODEL_ID = "jev-latest"
CONSOLIDATION_EVALUATION_BATCH_SIZE = 10

MERGE_SYSTEM_PROMPT = """
Combine the supplied memories into one self-contained durable fact.

Use only details explicitly supported by the supplied memories. Preserve every
compatible detail that will matter for future recall. Do not infer a broader
preference, habit, personality trait, or general pattern. Do not mention memory
IDs or the act of merging.

Return only valid JSON, with no Markdown, in this shape:
{"text": "One atomic merged memory."}
""".strip()


def discover_consolidation_pairs(
    seed_rows: list[tuple],
    candidate_limit: int,
    evaluated_pair_keys: set[tuple[int, int]] | None = None,
) -> list[tuple]:
    pairs = []
    seen_pair_keys = set(evaluated_pair_keys or set())

    for seed_row in seed_rows:
        candidates = find_consolidation_candidates(
            seed_row,
            limit=candidate_limit,
        )

        for candidate_row in candidates:
            pair_key = tuple(sorted(
                (seed_row[0], candidate_row[0])
            ))

            if pair_key in seen_pair_keys:
                continue

            seen_pair_keys.add(pair_key)
            pairs.append((seed_row, candidate_row))

    return pairs


def parse_merged_memory_text(raw_response: str) -> str:
    try:
        data = json.loads(raw_response)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("Merge generator returned invalid JSON") from error

    if not isinstance(data, dict):
        raise ValueError("Merge generator response must be an object")

    text = data.get("text")

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Merged memory text must be non-empty")

    return text.strip()


def generate_merged_memory_text(memories: list[dict]) -> str:
    response = request_chat_completion(
        base_url=BASE_URL,
        api_key=API_KEY,
        model_id=MODEL_ID,
        messages=[
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({
                    "memories": [
                        {
                            "memory_id": memory["id"],
                            "text": memory["text"],
                        }
                        for memory in memories
                    ],
                }),
            },
        ],
    )

    try:
        raw_text = response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(
            "Merge generator returned no message content"
        ) from error

    return parse_merged_memory_text(raw_text)


def merge_memory_entities(memories: list[dict]) -> list[dict]:
    merged = []
    seen_names = set()

    for memory in memories:
        for entity in memory["entities"]:
            normalized_name = " ".join(
                entity["name"].casefold().split()
            )

            if normalized_name in seen_names:
                continue

            seen_names.add(normalized_name)
            merged.append({
                "name": entity["name"],
                "context": entity["context"],
            })

    return merged


def prepare_merge(
    decision_id: int,
    candidate_limit: int = 10,
) -> dict | None:
    decision = load_consolidation_decision(decision_id)

    if decision["applied_action"] != "PENDING":
        raise ValueError(
            f"Decision {decision_id} is already applied"
        )

    if choose_consolidation_action(decision) != "MERGED":
        raise ValueError(
            f"Decision {decision_id} does not authorize a merge"
        )

    memories = load_pair_memory_snapshots(decision)

    if any(memory["status"] != "ACTIVE" for memory in memories):
        return None

    merged_text = generate_merged_memory_text(memories)
    merged_entities = merge_memory_entities(memories)
    merged_embedding = embed_text(merged_text)
    source_ids = {memory["id"] for memory in memories}

    write_candidates = find_write_candidates(
        merged_text,
        merged_entities,
        merged_embedding,
        limit=candidate_limit,
        exclude_memory_ids=source_ids,
    )
    relation_decisions = evaluate_memory_relations(
        merged_text,
        merged_entities,
        write_candidates,
    )

    return {
        "text": merged_text,
        "entities": merged_entities,
        "embedding": merged_embedding,
        "relation_decisions": relation_decisions,
    }


def apply_consolidation_decision(
    decision: dict,
    candidate_limit: int = 10,
) -> dict:
    action = choose_consolidation_action(decision)

    if action == "MERGED":
        prepared_merge = prepare_merge(
            decision["id"],
            candidate_limit=candidate_limit,
        )
        return apply_merge_decision(
            decision["id"],
            prepared_merge or {},
        )

    return apply_non_merge_decision(decision["id"])


def _chunks(items: list, chunk_size: int):
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def run_consolidation_batch(
    *,
    batch_limit: int = 20,
    candidate_limit: int = 10,
    decision_version: str = CONSOLIDATION_DECISION_VERSION,
    evaluation_batch_size: int = (
        CONSOLIDATION_EVALUATION_BATCH_SIZE
    ),
    apply_actions: bool = True,
) -> dict:
    for name, value in (
        ("batch_limit", batch_limit),
        ("candidate_limit", candidate_limit),
        ("evaluation_batch_size", evaluation_batch_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if not isinstance(decision_version, str) or not decision_version:
        raise ValueError("decision_version must be non-empty text")

    run_id = create_consolidation_run(
        decision_version,
        batch_limit,
        candidate_limit,
    )
    seed_count = 0
    proposed_pair_count = 0
    evaluated_pair_count = 0
    applied_action_count = 0

    try:
        seeds = load_consolidation_seeds(
            decision_version,
            batch_limit,
        )
        seed_ids = [seed[0] for seed in seeds]
        seed_count = len(seeds)
        record_consolidation_run_seeds(run_id, seed_ids)

        evaluated_pair_keys = load_evaluated_pair_keys(
            decision_version
        )
        pairs = discover_consolidation_pairs(
            seeds,
            candidate_limit,
            evaluated_pair_keys,
        )
        proposed_pair_count = len(pairs)

        for pair_batch in _chunks(
            pairs,
            evaluation_batch_size,
        ):
            decisions = evaluate_consolidation_pairs(pair_batch)
            decision_ids = save_consolidation_decisions(
                run_id,
                decision_version,
                CONSOLIDATION_MODEL_ID,
                decisions,
            )
            evaluated_pair_count += len(decision_ids)

        action_results = []
        action_failures = []

        if apply_actions:
            pending_decisions = (
                load_pending_consolidation_decisions(
                    decision_version
                )
            )

            for decision in pending_decisions:
                try:
                    result = apply_consolidation_decision(
                        decision,
                        candidate_limit=candidate_limit,
                    )
                    action_results.append(result)

                    if result["action"] not in {
                        "NONE",
                        "DEFERRED",
                    }:
                        applied_action_count += 1
                except Exception as error:
                    record_consolidation_decision_error(
                        decision["id"],
                        error,
                    )
                    action_failures.append({
                        "decision_id": decision["id"],
                        "error": str(error),
                    })

        if action_failures:
            error_message = (
                f"{len(action_failures)} consolidation action(s) "
                "remain pending"
            )
            fail_consolidation_run(
                run_id,
                error_message,
                seed_count=seed_count,
                proposed_pair_count=proposed_pair_count,
                evaluated_pair_count=evaluated_pair_count,
                applied_action_count=applied_action_count,
            )
            return {
                "run_id": run_id,
                "status": "FAILED",
                "decision_version": decision_version,
                "seed_count": seed_count,
                "proposed_pair_count": proposed_pair_count,
                "evaluated_pair_count": evaluated_pair_count,
                "applied_action_count": applied_action_count,
                "action_results": action_results,
                "action_failures": action_failures,
                "remaining_seed_count": (
                    count_pending_consolidation_seeds(
                        decision_version
                    )
                ),
            }

        mark_consolidation_seeds_processed(
            run_id,
            seed_ids,
            decision_version,
        )
        finish_consolidation_run(
            run_id,
            seed_count=seed_count,
            proposed_pair_count=proposed_pair_count,
            evaluated_pair_count=evaluated_pair_count,
            applied_action_count=applied_action_count,
        )

        return {
            "run_id": run_id,
            "status": "COMPLETED",
            "decision_version": decision_version,
            "seed_count": seed_count,
            "proposed_pair_count": proposed_pair_count,
            "evaluated_pair_count": evaluated_pair_count,
            "applied_action_count": applied_action_count,
            "action_results": action_results,
            "action_failures": [],
            "remaining_seed_count": (
                count_pending_consolidation_seeds(
                    decision_version
                )
            ),
        }

    except Exception as error:
        fail_consolidation_run(
            run_id,
            str(error),
            seed_count=seed_count,
            proposed_pair_count=proposed_pair_count,
            evaluated_pair_count=evaluated_pair_count,
            applied_action_count=applied_action_count,
        )
        raise
