import unittest
from unittest.mock import patch

from psycopg.types.json import Jsonb

import agent_harness.memory_consolidation as consolidation
from agent_harness.memory_consolidation import (
    merge_memory_entities,
    parse_merged_memory_text,
)
from agent_harness.memory_lifecycle import (
    _deactivate_memory,
    apply_merge_decision,
    choose_consolidation_action,
)
from agent_harness.storage.connection import connect_db


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

    def test_merge_output_is_strict_json(self):
        self.assertEqual(
            parse_merged_memory_text('{"text": "Merged fact."}'),
            "Merged fact.",
        )

        with self.assertRaises(ValueError):
            parse_merged_memory_text("Merged fact.")

    def test_entities_are_deduplicated_by_name(self):
        memories = [
            {
                "entities": [{
                    "name": "PostgreSQL",
                    "context": "Database",
                }],
            },
            {
                "entities": [{
                    "name": " postgresql ",
                    "context": "Backend",
                }],
            },
        ]

        self.assertEqual(
            merge_memory_entities(memories),
            [{"name": "PostgreSQL", "context": "Database"}],
        )


class ConsolidationBatchTests(unittest.TestCase):
    def test_decide_only_batch_persists_and_marks_seed(self):
        seed = (1, "seed", [], None, None)
        pair = (seed, (2, "candidate", [], None, None))
        decision = {"current_memory_id": 1}

        with (
            patch.object(
                consolidation,
                "create_consolidation_run",
                return_value=42,
            ),
            patch.object(
                consolidation,
                "load_consolidation_seeds",
                return_value=[seed],
            ),
            patch.object(
                consolidation,
                "record_consolidation_run_seeds",
            ) as record_seeds,
            patch.object(
                consolidation,
                "load_evaluated_pair_keys",
                return_value=set(),
            ),
            patch.object(
                consolidation,
                "discover_consolidation_pairs",
                return_value=[pair],
            ),
            patch.object(
                consolidation,
                "evaluate_consolidation_pairs",
                return_value=[decision],
            ),
            patch.object(
                consolidation,
                "save_consolidation_decisions",
                return_value=[7],
            ) as save_decisions,
            patch.object(
                consolidation,
                "mark_consolidation_seeds_processed",
            ) as mark_processed,
            patch.object(
                consolidation,
                "finish_consolidation_run",
            ) as finish_run,
            patch.object(
                consolidation,
                "count_pending_consolidation_seeds",
                return_value=0,
            ),
        ):
            report = consolidation.run_consolidation_batch(
                batch_limit=1,
                candidate_limit=1,
                apply_actions=False,
            )

        self.assertEqual(report["status"], "COMPLETED")
        self.assertEqual(report["evaluated_pair_count"], 1)
        self.assertEqual(report["remaining_seed_count"], 0)
        record_seeds.assert_called_once_with(42, [1])
        save_decisions.assert_called_once()
        mark_processed.assert_called_once_with(42, [1], "v1")
        finish_run.assert_called_once()


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

    def test_merge_inserts_result_before_deactivating_sources(self):
        canonical_id, duplicate_id, _neighbor_id, _earlier_id = (
            self.memory_ids
        )
        memory_a_id, memory_b_id = sorted(
            (canonical_id, duplicate_id)
        )
        merge_decision_id = self.conn.execute(
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
                0.0, 0.0, 0.0, 0.0,
                'merge', 0.95, %s, 'merge-test', 'test'
            )
            RETURNING id
            """,
            (
                memory_a_id,
                memory_b_id,
                canonical_id,
                duplicate_id,
                Jsonb({
                    "keep_separate": 0.03,
                    "merge": 0.95,
                    "uncertain": 0.02,
                }),
            ),
        ).fetchone()[0]
        self.conn.commit()
        merged_id = None

        try:
            result = apply_merge_decision(
                merge_decision_id,
                {
                    "text": "merged result",
                    "entities": [],
                    "embedding": [0.0] * 2048,
                    "relation_decisions": [],
                },
            )

            merged_id = result["result_memory_id"]
            merged = self.conn.execute(
                """
                SELECT status, origin_type, source_message_id
                FROM memories
                WHERE id = %s
                """,
                (merged_id,),
            ).fetchone()
            self.assertEqual(merged, ("ACTIVE", "MERGED", None))

            source_rows = self.conn.execute(
                """
                SELECT status, replaced_by_memory_id
                FROM memories
                WHERE id = ANY(%s)
                ORDER BY id
                """,
                ([canonical_id, duplicate_id],),
            ).fetchall()
            self.assertEqual(
                source_rows,
                [
                    ("MERGED_SOURCE", merged_id),
                    ("MERGED_SOURCE", merged_id),
                ],
            )
        finally:
            cleanup_ids = list(self.memory_ids)

            if merged_id is not None:
                cleanup_ids.append(merged_id)

            cleanup_conn = connect_db()
            cleanup_conn.execute(
                "DELETE FROM memory_lifecycle_edges WHERE decision_id IN (%s, %s)",
                (self.decision_id, merge_decision_id),
            )
            cleanup_conn.execute(
                "DELETE FROM memory_edges WHERE source_memory_id = ANY(%s) OR target_memory_id = ANY(%s)",
                (cleanup_ids, cleanup_ids),
            )
            cleanup_conn.execute(
                "DELETE FROM memory_consolidation_decisions WHERE id IN (%s, %s)",
                (self.decision_id, merge_decision_id),
            )
            cleanup_conn.execute(
                "DELETE FROM memories WHERE id = ANY(%s)",
                (cleanup_ids,),
            )
            cleanup_conn.commit()
            cleanup_conn.close()
            self.conn.close()
            self.conn = connect_db()


if __name__ == "__main__":
    unittest.main()
