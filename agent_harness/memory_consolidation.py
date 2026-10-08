from agent_harness.db import find_consolidation_candidates


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
            pairs.append(
                (seed_row, candidate_row)
            )

    return pairs
