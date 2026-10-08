from agent_harness.db import (
    find_query_anchors,
    load_graph_neighbors,
    select_graph_expansion_candidates,
)
from agent_harness.embedding_client import embed_text
from agent_harness.jev_client import (
    activate_query_graphs,
    allocate_query_graph_budgets,
    calculate_query_depth_limit,
    evaluate_query_routing,
    evaluate_evidence_sufficiency,
    is_evidence_sufficient,
)


def retrieve_memory_evidence(
    query: str,
    anchor_limit: int = 10,
) -> dict:
    routing = evaluate_query_routing(query)

    active_graphs = activate_query_graphs(routing)

    graph_budgets = allocate_query_graph_budgets(
        active_graphs
    )

    depth_limit = calculate_query_depth_limit(routing)

    query_embedding = embed_text(query)

    anchors = find_query_anchors(
        query,
        query_embedding,
        limit=anchor_limit,
    )

    anchor_ids = {
        row[0]
        for row in anchors
    }

    retrieval = {
        "routing": routing,
        "active_graphs": active_graphs,
        "graph_budgets": graph_budgets,
        "graph_budget_used": {
            graph: 0
            for graph in graph_budgets
        },
        "depth_limit": depth_limit,
        "depth": 0,
        "evidence": anchors,
        "visited_ids": anchor_ids,
        "frontier_ids": anchor_ids,
    }

    if not anchors:
        retrieval["assessment"] = None
        retrieval["evidence_sufficient"] = False
        retrieval["stop_reason"] = "no_anchors"
        return retrieval

    assessment = evaluate_evidence_sufficiency(
        query,
        anchors,
        depth=0,
    )

    retrieval["assessment"] = assessment
    retrieval["evidence_sufficient"] = (
        is_evidence_sufficient(assessment)
    )

    while not retrieval["evidence_sufficient"]:
        if retrieval["depth"] >= depth_limit:
            retrieval["stop_reason"] = (
                "depth_limit_reached"
            )
            return retrieval

        budgets_exhausted = all(
            retrieval["graph_budget_used"][graph]
            >= graph_budgets[graph]
            for graph in graph_budgets
        )

        if budgets_exhausted:
            retrieval["stop_reason"] = (
                "graph_budgets_exhausted"
            )
            return retrieval

        neighbor_rows = load_graph_neighbors(
            frontier_ids=retrieval["frontier_ids"],
            active_graphs=active_graphs,
        )

        selected_rows, updated_usage = (
            select_graph_expansion_candidates(
                neighbor_rows=neighbor_rows,
                visited_ids=retrieval["visited_ids"],
                graph_budgets=graph_budgets,
                graph_budget_used=(
                    retrieval["graph_budget_used"]
                ),
            )
        )

        if not selected_rows:
            retrieval["stop_reason"] = (
                "no_new_neighbors"
            )
            return retrieval

        selected_ids = {
            row[0]
            for row in selected_rows
        }

        retrieval["evidence"].extend(
            selected_rows
        )

        retrieval["visited_ids"].update(
            selected_ids
        )

        retrieval["frontier_ids"] = selected_ids
        retrieval["graph_budget_used"] = (
            updated_usage
        )

        retrieval["depth"] += 1

        assessment = evaluate_evidence_sufficiency(
            query,
            retrieval["evidence"],
            depth=retrieval["depth"],
        )

        retrieval["assessment"] = assessment
        retrieval["evidence_sufficient"] = (
            is_evidence_sufficient(assessment)
        )

    retrieval["stop_reason"] = (
        "evidence_sufficient"
    )

    return retrieval