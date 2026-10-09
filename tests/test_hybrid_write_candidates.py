import unittest
from unittest.mock import patch

from agent_harness.storage.memories import find_write_candidates


class HybridWriteCandidateTests(unittest.TestCase):
    def test_fuses_vector_and_bm25_rankings_before_selecting_candidates(self):
        vector_rows = [
            (1, "vector first"),
            (2, "vector second"),
            (3, "vector third"),
        ]
        bm25_matches = [
            ((2, "bm25 first"), 5.0),
            ((3, "bm25 second"), 4.0),
            ((4, "bm25 third"), 3.0),
        ]
        rows_by_id = {
            1: (1, "memory 1"),
            2: (2, "memory 2"),
            3: (3, "memory 3"),
            4: (4, "memory 4"),
        }

        with (
            patch(
                "agent_harness.storage.memories.search_vector_candidates",
                return_value=vector_rows,
            ),
            patch(
                "agent_harness.storage.memories.search_bm25_candidates",
                return_value=bm25_matches,
            ),
            patch(
                "agent_harness.storage.memories.load_candidates_by_ids",
                side_effect=lambda ids, _embedding: [
                    rows_by_id[memory_id]
                    for memory_id in ids
                ],
            ),
        ):
            candidates = find_write_candidates(
                "new memory",
                [0.1, 0.2],
                limit=3,
            )

        self.assertEqual(
            [row[0] for row in candidates],
            [2, 3, 1],
        )

    def test_excluded_memories_are_removed_after_hybrid_ranking(self):
        vector_rows = [(1, "first"), (2, "second")]
        bm25_matches = [((2, "second"), 2.0), ((3, "third"), 1.0)]
        rows_by_id = {
            1: (1, "memory 1"),
            2: (2, "memory 2"),
            3: (3, "memory 3"),
        }

        with (
            patch(
                "agent_harness.storage.memories.search_vector_candidates",
                return_value=vector_rows,
            ),
            patch(
                "agent_harness.storage.memories.search_bm25_candidates",
                return_value=bm25_matches,
            ),
            patch(
                "agent_harness.storage.memories.load_candidates_by_ids",
                side_effect=lambda ids, _embedding: [
                    rows_by_id[memory_id]
                    for memory_id in ids
                ],
            ),
        ):
            candidates = find_write_candidates(
                "new memory",
                [0.1, 0.2],
                limit=10,
                exclude_memory_id=2,
                exclude_memory_ids={3},
            )

        self.assertEqual([row[0] for row in candidates], [1])
