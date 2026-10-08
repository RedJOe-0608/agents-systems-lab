import argparse
import json

from agent_harness.memory_consolidation import (
    CONSOLIDATION_DECISION_VERSION,
    run_consolidation_batch,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one bounded memory-consolidation batch."
    )
    parser.add_argument("--batch-limit", type=int, default=20)
    parser.add_argument("--candidate-limit", type=int, default=10)
    parser.add_argument(
        "--decision-version",
        default=CONSOLIDATION_DECISION_VERSION,
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--decide-only",
        action="store_true",
        help=(
            "Persist Jev decisions but leave lifecycle actions "
            "pending for a later normal run."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    report = run_consolidation_batch(
        batch_limit=args.batch_limit,
        candidate_limit=args.candidate_limit,
        decision_version=args.decision_version,
        evaluation_batch_size=args.evaluation_batch_size,
        apply_actions=not args.decide_only,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
