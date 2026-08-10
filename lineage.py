"""
Module 3 & 4: Cell Events & Lineage Graph Analyzer.
Parses tracking DiGraph to detect mitosis (division), apoptosis (death), appearance events,
and constructs multi-generational lineage family trees.
"""

from typing import Dict, List, Any, Tuple
import networkx as nx
import pandas as pd


def detect_cell_events(track_graph: nx.DiGraph, total_frames: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Parses tracking graph to identify cell division (mitosis), death, and appearance events.
    """
    events = []
    mitosis_count = 0
    death_count = 0
    appearance_count = 0
    continuity_count = 0

    for node in track_graph.nodes():
        # Node format in Trackastra is typically (t, label_id) or string representation
        if isinstance(node, tuple) and len(node) >= 2:
            frame, label_id = node[0], node[1]
        else:
            frame, label_id = None, str(node)

        in_edges = list(track_graph.in_edges(node))
        out_edges = list(track_graph.out_edges(node))

        in_deg = len(in_edges)
        out_deg = len(out_edges)

        event_type = "continuity"

        # Mitosis: 1 cell splits into 2 or more daughter cells
        if out_deg >= 2:
            event_type = "division"
            mitosis_count += 1

        # Death: Cell disappears before final frame
        elif out_deg == 0 and frame is not None and frame < total_frames - 1:
            event_type = "death"
            death_count += 1

        # Appearance: New cell enters field after frame 0 without parent
        elif in_deg == 0 and frame is not None and frame > 0:
            event_type = "appearance"
            appearance_count += 1

        else:
            continuity_count += 1

        events.append({
            "node": str(node),
            "frame": frame,
            "label_id": label_id,
            "event_type": event_type,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "parents": [str(u) for u, v in in_edges],
            "children": [str(v) for u, v in out_edges]
        })

    df_events = pd.DataFrame(events)
    summary = {
        "division_events": mitosis_count,
        "death_events": death_count,
        "appearance_events": appearance_count,
        "continuity_events": continuity_count,
        "total_graph_nodes": track_graph.number_of_nodes(),
        "total_graph_edges": track_graph.number_of_edges()
    }

    print(f"[Lineage] Graph analysis: {mitosis_count} divisions, {death_count} deaths, {appearance_count} appearances.")
    return df_events, summary


def build_lineage_family_trees(track_graph: nx.DiGraph) -> List[Dict[str, Any]]:
    """
    Identifies root ancestor cells and extracts complete multi-generational family lineage trees.
    """
    root_nodes = [node for node, in_deg in track_graph.in_degree() if in_deg == 0]
    family_trees = []

    for root in root_nodes:
        # Get all descendant nodes in the DAG family tree
        descendants = nx.descendants(track_graph, root)
        family_members = {root} | descendants
        tree_subgraph = track_graph.subgraph(family_members)

        # Count divisions inside this lineage family
        divisions = sum(1 for n in tree_subgraph.nodes() if tree_subgraph.out_degree(n) >= 2)

        family_trees.append({
            "root_cell": str(root),
            "family_size": len(family_members),
            "division_count": divisions,
            "max_generational_depth": nx.dag_longest_path_length(tree_subgraph) if len(family_members) > 1 else 0
        })

    print(f"[Lineage] Extracted {len(family_trees)} distinct cell lineage family trees.")
    return family_trees


if __name__ == "__main__":
    # Test lineage analysis with a dummy DiGraph
    G = nx.DiGraph()
    G.add_edge((0, 1), (1, 1))
    G.add_edge((1, 1), (2, 1))
    G.add_edge((1, 1), (2, 2))  # Division at (1, 1) into (2, 1) and (2, 2)

    df_ev, sum_stats = detect_cell_events(G, total_frames=3)
    trees = build_lineage_family_trees(G)
    print("Summary:", sum_stats)
    print("Trees:", trees)
