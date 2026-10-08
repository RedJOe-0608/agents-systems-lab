import unittest

from psycopg.types.json import Jsonb

from agent_harness.db import connect_db
from agent_harness.memory_consolidation import (
    _deactivate_memory,
    choose_consolidation_action,
)


class ConsolidationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.decision = {
            "contradiction_score": 0.0,
            "current_supersedes_candidate_score": 0.0,
            "candidate_supersedes_current_score": 0.0,
            "redundancy_score": 0.0,
            "representation": "keep_separate",
            "representation_probability": 0.9,
        }

    def test_contradiction_blocks_other_actions(self):
        decision = {
            **self.decision,
            "contradiction_score": 0.9,
            "redundancy_score": 0.99,
            "representation": "merge",
        }

        self.assertEqual(
            choose_consolidation_action(decision),
            "CONTRADICTION_LINKED",
        )

    def test_ambiguous_supersession_is_deferred(self):
        decision = {
            **self.decision,
            "current_supersedes_candidate_score": 0.9,
            "candidate_supersedes_current_score": 0.9,
        }

        self.assertEqual(
            choose_consolidation_action(decision),
            "DEFERRED",
        )

    def test_merge_requires_selected_choice_above_threshold(self):
        decision = {
            **self.decision,
            "representation": "merge",
            "representation_probability": 0.85,
        }

        self.assertEqual(
            choose_consolidation_action(decision),
            "MERGED",
        )


class ReplacementTransactionTests(unittest.TestCase):
    def setUp(self):
        self.conn = connect_db()

        self.memory_ids = []
        for text in (
            "canonical",
            "duplicate",
            "neighbor",
            "earlier duplicate",
        ):
            memory_id = self.conn.execute(
                """
                INSERT INTO memories (
                    text,
                    source_message_id,
                    entities,
                    origin_type
                )
                VALUES (%s, NULL, '[]'::jsonb, 'MERGED')
                RETURNING id
                """,
                (text,),
            ).fetchone()[0]
            self.memory_ids.append(memory_id)

        canonical_id, duplicate_id, neighbor_id, earlier_id = (
            self.memory_ids
        )
        memory_a_id, memory_b_id = sorted(
            (canonical_id, duplicate_id)
        )

        self.decision_id = self.conn.execute(
            """
            INSERT INTO memory_consolidation_decisions (
                memory_a_id,
                memory_b_id,
                current_memory_id,
                candidate_memory_id,
                redundancy_score,
                contradiction_score,
                current_supersedes_candidate_score,
                candidate_supersedes_current_score,
                representation,
                representation_probability,
                representation_probabilities,
                decision_version,
                model_id
            )
            VALUES (
                %s, %s, %s, %s,
                0.95, 0.0, 0.0, 0.0,
                'keep_separate', 0.95, %s, 'test', 'test'
            )
            RETURNING id
            """,
            (
                memory_a_id,
                memory_b_id,
                canonical_id,
                duplicate_id,
                Jsonb({
                    "keep_separate": 0.95,
                    "merge": 0.03,
                    "uncertain": 0.02,
                }),
            ),
        ).fetchone()[0]

        source_id, target_id = sorted(
            (duplicate_id, neighbor_id)
        )
        self.conn.execute(
            """
            INSERT INTO memory_edges (
                source_memory_id,
                target_memory_id,
                relation_type,
                score
            )
            VALUES (%s, %s, 'RELATED_TO', 0.8)
            """,
            (source_id, target_id),
        )
        self.conn.execute(
            """
            INSERT INTO memory_lifecycle_edges (
                source_memory_id,
                target_memory_id,
                relation_type,
                score,
                decision_id
            )
            VALUES (%s, %s, 'CONTRADICTS', 0.9, %s)
            """,
            (source_id, target_id, self.decision_id),
        )
        self.conn.execute(
            """
            UPDATE memories
            SET status = 'REDUNDANT',
                replaced_by_memory_id = %s
            WHERE id = %s
            """,
            (duplicate_id, earlier_id),
        )

    def tearDown(self):
        self.conn.rollback()
        self.conn.close()

    def test_deactivation_migrates_edges_and_collapses_pointers(self):
        canonical_id, duplicate_id, neighbor_id, earlier_id = (
            self.memory_ids
        )

        _deactivate_memory(
            self.conn,
            inactive_memory_id=duplicate_id,
            replacement_memory_id=canonical_id,
            status="REDUNDANT",
            relation_type="REDUNDANT_OF",
            score=0.95,
            decision_id=self.decision_id,
        )

        status, replacement_id = self.conn.execute(
            """
            SELECT status, replaced_by_memory_id
            FROM memories
            WHERE id = %s
            """,
            (duplicate_id,),
        ).fetchone()
        self.assertEqual(status, "REDUNDANT")
        self.assertEqual(replacement_id, canonical_id)

        earlier_replacement = self.conn.execute(
            """
            SELECT replaced_by_memory_id
            FROM memories
            WHERE id = %s
            """,
            (earlier_id,),
        ).fetchone()[0]
        self.assertEqual(earlier_replacement, canonical_id)

        source_id, target_id = sorted(
            (canonical_id, neighbor_id)
        )
        migrated_edge = self.conn.execute(
            """
            SELECT score
            FROM memory_edges
            WHERE source_memory_id = %s
              AND target_memory_id = %s
              AND relation_type = 'RELATED_TO'
            """,
            (source_id, target_id),
        ).fetchone()
        self.assertIsNotNone(migrated_edge)

        rewired_contradiction = self.conn.execute(
            """
            SELECT score
            FROM memory_lifecycle_edges
            WHERE source_memory_id = %s
              AND target_memory_id = %s
              AND relation_type = 'CONTRADICTS'
            """,
            (source_id, target_id),
        ).fetchone()
        self.assertIsNotNone(rewired_contradiction)


if __name__ == "__main__":
    unittest.main()
