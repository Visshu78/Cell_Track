"""
Lineage Tree Visualizer Generator.
Parses NetworkX cell tracking graph and builds a self-contained, interactive 
D3.js pedigree family tree and Gantt timeline visualizer (lineage_tree_viewer.html).
"""

import json
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
from data_loader import load_masks
from model_loader import get_trackastra_model
from tracker import run_cell_tracking
from lineage import detect_cell_events, build_lineage_family_trees
from morphology import extract_dataset_morphology
from behavior import compute_cell_kinematics


def parse_lineage_graph_to_json(track_graph: nx.DiGraph, df_kinematics: pd.DataFrame = None) -> dict:
    """
    Parses tracking DiGraph into hierarchical JSON family structures for D3.js tree rendering.
    """
    # Identify root ancestors (in-degree == 0)
    root_nodes = [node for node, in_deg in track_graph.in_degree() if in_deg == 0]
    
    # Map kinematics by cell_id/label_id if available
    kinematics_map = {}
    if df_kinematics is not None and not df_kinematics.empty:
        id_col = "label_id" if "label_id" in df_kinematics.columns else "cell_id"
        for _, row in df_kinematics.iterrows():
            cid = int(row[id_col])
            kinematics_map[cid] = {
                "mean_speed": round(float(row.get("mean_speed", 0.0)), 2),
                "net_displacement": round(float(row.get("net_displacement", 0.0)), 2),
                "directionality": round(float(row.get("directionality_ratio", row.get("directionality", 0.0))), 3)
            }

    # Group graph nodes by cell trajectory ID
    # In Trackastra graph, nodes are (t, label_id) tuples
    node_frames = {}
    cell_lifespans = {}
    cell_parents = {}
    cell_children = {}

    for node in track_graph.nodes():
        if isinstance(node, (tuple, list)) and len(node) >= 2:
            frame, c_id = int(node[0]), int(node[1])
        else:
            frame, c_id = 0, int(node)
            
        node_frames[node] = (frame, c_id)
        if c_id not in cell_lifespans:
            cell_lifespans[c_id] = {"start_frame": frame, "end_frame": frame, "nodes": [node]}
        else:
            cell_lifespans[c_id]["start_frame"] = min(cell_lifespans[c_id]["start_frame"], frame)
            cell_lifespans[c_id]["end_frame"] = max(cell_lifespans[c_id]["end_frame"], frame)
            cell_lifespans[c_id]["nodes"].append(node)

    # Detect cell-level parent/child transitions (mitosis or continuation)
    division_events = []
    for node in track_graph.nodes():
        out_edges = list(track_graph.out_edges(node))
        if len(out_edges) >= 2:
            # Division node!
            p_frame, p_id = node_frames[node]
            c_ids = [node_frames[target][1] for source, target in out_edges]
            division_events.append({
                "parent_id": p_id,
                "frame": p_frame,
                "daughters": list(set(c_ids))
            })
            for d_id in set(c_ids):
                cell_parents[d_id] = p_id
                if p_id not in cell_children:
                    cell_children[p_id] = [d_id]
                else:
                    if d_id not in cell_children[p_id]:
                        cell_children[p_id].append(d_id)

    # Build hierarchical tree structure for each family
    all_cell_ids = sorted(list(cell_lifespans.keys()))
    root_cell_ids = [c for c in all_cell_ids if c not in cell_parents]

    families = []
    for fam_idx, r_id in enumerate(root_cell_ids):
        def build_node_hierarchy(c_id, generation=0):
            ls = cell_lifespans[c_id]
            kin = kinematics_map.get(c_id, {"mean_speed": 0.0, "net_displacement": 0.0, "directionality": 0.0})
            children_ids = cell_children.get(c_id, [])
            
            node_data = {
                "id": c_id,
                "name": f"Cell #{c_id}",
                "generation": generation,
                "start_frame": ls["start_frame"],
                "end_frame": ls["end_frame"],
                "duration": ls["end_frame"] - ls["start_frame"] + 1,
                "parent": cell_parents.get(c_id, None),
                "speed": kin["mean_speed"],
                "displacement": kin["net_displacement"],
                "directionality": kin["directionality"],
                "has_division": len(children_ids) > 0,
                "children": [build_node_hierarchy(ch_id, generation + 1) for ch_id in children_ids]
            }
            return node_data

        fam_tree = build_node_hierarchy(r_id)
        families.append({
            "family_id": fam_idx + 1,
            "root_id": r_id,
            "tree": fam_tree
        })

    return {
        "total_cells": len(all_cell_ids),
        "total_families": len(families),
        "total_divisions": len(division_events),
        "divisions": division_events,
        "families": families,
        "cell_lifespans": [
            {
                "id": c_id,
                "start": ls["start_frame"],
                "end": ls["end_frame"],
                "parent": cell_parents.get(c_id, None),
                "children": cell_children.get(c_id, []),
                "speed": kinematics_map.get(c_id, {}).get("mean_speed", 0.0)
            }
            for c_id, ls in cell_lifespans.items()
        ]
    }


def build_lineage_web_visualizer():
    print("[LineageVisualizer] Executing tracking & extracting lineage hierarchy...")
    masks = load_masks()
    model = get_trackastra_model()
    tracked_masks, track_graph = run_cell_tracking(masks=masks, model=model)
    
    df_morphology = extract_dataset_morphology(tracked_masks)
    df_kinematics = compute_cell_kinematics(df_morphology)
    
    lineage_data = parse_lineage_graph_to_json(track_graph, df_kinematics)
    json_str = json.dumps(lineage_data, indent=2)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cell Track - Mitotic Lineage Tree Visualizer</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {{ font-family: 'Outfit', sans-serif; background-color: #0b0f19; color: #f1f5f9; }}
        .glass-panel {{ background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border: 1px solid rgba(255, 255, 255, 0.08); }}
        .node circle {{ fill: #38bdf8; stroke: #0284c7; stroke-width: 2px; transition: all 0.3s; }}
        .node.division circle {{ fill: #f59e0b; stroke: #d97706; stroke-width: 3px; }}
        .node text {{ font-size: 12px; font-weight: 600; fill: #f1f5f9; text-shadow: 0 1px 3px rgba(0,0,0,0.8); }}
        .link {{ fill: none; stroke: rgba(148, 163, 184, 0.4); stroke-width: 2px; stroke-dasharray: 4 2; }}
        .link.active {{ stroke: #38bdf8; stroke-width: 3px; stroke-dasharray: none; }}
        .gantt-bar {{ rx: 4; ry: 4; transition: all 0.2s; cursor: pointer; }}
        .gantt-bar:hover {{ filter: brightness(1.25); }}
    </style>
</head>
<body class="min-h-screen p-4 md:p-8 flex flex-col">
    <!-- Header -->
    <header class="max-w-7xl w-full mx-auto mb-6 flex flex-wrap items-center justify-between gap-4 glass-panel p-6 rounded-2xl">
        <div class="flex items-center gap-3">
            <div class="p-3 bg-amber-500/20 text-amber-400 rounded-xl border border-amber-500/30">
                🧬
            </div>
            <div>
                <h1 class="text-2xl font-bold text-white tracking-tight">Cell Mitotic Lineage Tree & Pedigree Viewer</h1>
                <p class="text-xs text-slate-400">Interactive D3.js Multi-Generational Family Trees, Mitosis Events & Lifespan Gantt Chart</p>
            </div>
        </div>
        <div class="flex items-center gap-3 text-sm">
            <span class="px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-400 border border-amber-500/20 font-medium">
                ● Interactive Pedigree Graph
            </span>
            <span class="px-3 py-1.5 rounded-lg bg-slate-800 text-slate-300 border border-slate-700 font-mono" id="stat-summary">
                Loading...
            </span>
        </div>
    </header>

    <!-- Main Grid -->
    <main class="max-w-7xl w-full mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1">
        <!-- Main Lineage Tree Canvas (2 cols) -->
        <div class="lg:col-span-2 flex flex-col gap-4 glass-panel p-6 rounded-2xl">
            <!-- View Selector Tabs -->
            <div class="flex items-center justify-between border-b border-slate-700/50 pb-4">
                <div class="flex items-center gap-2">
                    <button id="tab-tree" onclick="setTab('tree')" class="px-4 py-2 text-xs font-semibold rounded-lg bg-amber-500 text-slate-950 shadow-lg shadow-amber-500/20 transition">
                        🌳 D3.js Pedigree Family Trees
                    </button>
                    <button id="tab-gantt" onclick="setTab('gantt')" class="px-4 py-2 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700 transition">
                        📊 Lifespan Gantt Chart
                    </button>
                </div>
                <div class="flex items-center gap-2 text-xs text-slate-400">
                    <span>Family:</span>
                    <select id="family-select" onchange="renderActiveFamily()" class="bg-slate-900 border border-slate-700 text-cyan-300 px-3 py-1.5 rounded-lg font-medium outline-none">
                        <!-- Populated by JS -->
                    </select>
                </div>
            </div>

            <!-- Visualization Canvas -->
            <div id="canvas-container" class="relative w-full h-[520px] bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
                <svg id="lineageSvg" class="w-full h-full"></svg>
            </div>
        </div>

        <!-- Sidebar Inspector -->
        <div class="flex flex-col gap-6">
            <!-- Selected Cell Inspector Card -->
            <div class="glass-panel p-6 rounded-2xl">
                <h3 class="text-lg font-bold text-white mb-3 flex items-center gap-2">
                    🔍 Cell Pedigree Inspector
                </h3>
                <div id="inspector-content" class="space-y-3 text-xs text-slate-300">
                    <p class="text-slate-500 italic">Click on any node in the lineage tree or Gantt bar to inspect ancestral relationships and motility stats.</p>
                </div>
            </div>

            <!-- Lineage Statistics Card -->
            <div class="glass-panel p-6 rounded-2xl flex-1 flex flex-col">
                <h3 class="text-lg font-bold text-white mb-3 flex items-center gap-2">
                    📈 Mitosis & Lineage Summary
                </h3>
                <div class="space-y-3 text-xs text-slate-300">
                    <div class="flex justify-between py-1.5 border-b border-slate-700/50">
                        <span class="text-slate-400">Total Tracked Cells:</span>
                        <span class="font-mono text-cyan-400" id="stat-total-cells">0</span>
                    </div>
                    <div class="flex justify-between py-1.5 border-b border-slate-700/50">
                        <span class="text-slate-400">Root Ancestor Lineages:</span>
                        <span class="font-mono text-amber-400" id="stat-families">0</span>
                    </div>
                    <div class="flex justify-between py-1.5 border-b border-slate-700/50">
                        <span class="text-slate-400">Mitotic Division Events:</span>
                        <span class="font-mono text-emerald-400" id="stat-divisions">0</span>
                    </div>
                    <div class="flex justify-between py-1.5">
                        <span class="text-slate-400">Max Lineage Depth:</span>
                        <span class="font-mono" id="stat-depth">0 Generations</span>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Data & D3 Rendering Script -->
    <script>
        const lineageData = {json_str};
        let activeTab = 'tree';
        let activeFamilyIndex = 0;

        // Initialize Header Stats
        document.getElementById('stat-total-cells').innerText = lineageData.total_cells;
        document.getElementById('stat-families').innerText = lineageData.total_families;
        document.getElementById('stat-divisions').innerText = lineageData.total_divisions;
        document.getElementById('stat-summary').innerText = lineageData.total_cells + " Cells | " + lineageData.total_families + " Families | " + lineageData.total_divisions + " Mitosis Events";

        // Populate Family Selector
        const selectEl = document.getElementById('family-select');
        selectEl.innerHTML = '<option value="all">All Families (Overview)</option>';
        lineageData.families.forEach((fam, idx) => {{
            const opt = document.createElement('option');
            opt.value = idx;
            opt.innerText = "Family #" + fam.family_id + " (Root Cell #" + fam.root_id + ")";
            selectEl.appendChild(opt);
        }});

        function setTab(tab) {{
            activeTab = tab;
            document.getElementById('tab-tree').className = tab === 'tree' 
                ? "px-4 py-2 text-xs font-semibold rounded-lg bg-amber-500 text-slate-950 shadow-lg shadow-amber-500/20 transition"
                : "px-4 py-2 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700 transition";
            document.getElementById('tab-gantt').className = tab === 'gantt' 
                ? "px-4 py-2 text-xs font-semibold rounded-lg bg-amber-500 text-slate-950 shadow-lg shadow-amber-500/20 transition"
                : "px-4 py-2 text-xs font-semibold rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700 transition";
            renderActiveFamily();
        }}

        function renderActiveFamily() {{
            const val = selectEl.value;
            const container = document.getElementById('canvas-container');
            const svg = d3.select("#lineageSvg");
            svg.selectAll("*").remove();

            const width = container.clientWidth;
            const height = container.clientHeight;

            if (activeTab === 'tree') {{
                renderD3Tree(val, svg, width, height);
            }} else {{
                renderGanttChart(val, svg, width, height);
            }}
        }}

        function renderD3Tree(famVal, svg, width, height) {{
            const margin = {{top: 40, right: 60, bottom: 40, left: 80}};
            const innerW = width - margin.left - margin.right;
            const innerH = height - margin.top - margin.bottom;

            const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);

            let rootData;
            if (famVal === 'all' || lineageData.families.length === 0) {{
                // Virtual root for all families
                rootData = {{
                    name: "Root",
                    children: lineageData.families.map(f => f.tree)
                }};
            }} else {{
                rootData = lineageData.families[parseInt(famVal)].tree;
            }}

            const root = d3.hierarchy(rootData);
            const treeLayout = d3.tree().size([innerH, innerW]);
            treeLayout(root);

            // Links
            g.selectAll(".link")
                .data(root.links())
                .enter().append("path")
                .attr("class", "link")
                .attr("d", d3.linkHorizontal().x(d => d.y).y(d => d.x));

            # Nodes
            const node = g.selectAll(".node")
                .data(root.descendants())
                .enter().append("g")
                .attr("class", d => "node " + (d.data.has_division ? "division" : ""))
                .attr("transform", d => `translate(${{d.y}},${{d.x}})`)
                .on("click", (event, d) => inspectCellNode(d.data));

            node.append("circle")
                .attr("r", d => d.data.has_division ? 9 : 6);

            node.append("text")
                .attr("dy", ".35em")
                .attr("x", d => d.children ? -12 : 12)
                .attr("text-anchor", d => d.children ? "end" : "start")
                .text(d => d.data.id ? "#" + d.data.id : "");
        }}

        function renderGanttChart(famVal, svg, width, height) {{
            const margin = {{top: 40, right: 40, bottom: 50, left: 70}};
            const innerW = width - margin.left - margin.right;
            const innerH = height - margin.top - margin.bottom;

            const g = svg.append("g").attr("transform", `translate(${{margin.left}},${{margin.top}})`);

            let cellsToRender = lineageData.cell_lifespans;
            if (famVal !== 'all') {{
                const rootId = lineageData.families[parseInt(famVal)].root_id;
                // Filter cells in this family
                const famCellIds = new Set();
                function collect(node) {{
                    famCellIds.add(node.id);
                    if(node.children) node.children.forEach(collect);
                }}
                collect(lineageData.families[parseInt(famVal)].tree);
                cellsToRender = cellsToRender.filter(c => famCellIds.has(c.id));
            }}

            const xScale = d3.scaleLinear().domain([0, 29]).range([0, innerW]);
            const yScale = d3.scaleBand().domain(cellsToRender.map(c => "Cell #" + c.id)).range([0, innerH]).padding(0.25);

            // Axes
            g.append("g")
                .attr("transform", `translate(0,${{innerH}})`)
                .call(d3.axisBottom(xScale).ticks(10).tickFormat(d => "Frame " + d))
                .attr("color", "#94a3b8");

            g.append("g")
                .call(d3.axisLeft(yScale))
                .attr("color", "#94a3b8");

            // Gantt Bars
            g.selectAll(".gantt-bar")
                .data(cellsToRender)
                .enter().append("rect")
                .attr("class", "gantt-bar")
                .attr("x", d => xScale(d.start))
                .attr("y", d => yScale("Cell #" + d.id))
                .attr("width", d => Math.max(8, xScale(d.end) - xScale(d.start)))
                .attr("height", yScale.bandwidth())
                .attr("fill", d => d.children && d.children.length > 0 ? "#f59e0b" : "#38bdf8")
                .on("click", (event, d) => inspectCellById(d.id));
        }}

        function inspectCellNode(nodeData) {{
            if(!nodeData || !nodeData.id) return;
            const el = document.getElementById('inspector-content');
            el.innerHTML = `
                <div class="p-3 bg-slate-900 rounded-xl border border-slate-800 space-y-2">
                    <div class="flex items-center justify-between border-b border-slate-800 pb-2">
                        <span class="text-sm font-bold text-amber-400">Cell #${{nodeData.id}}</span>
                        <span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono text-[10px]">Gen ${{nodeData.generation || 0}}</span>
                    </div>
                    <div class="grid grid-cols-2 gap-2 text-[11px]">
                        <div><span class="text-slate-400">Lifespan:</span> <span class="font-mono text-cyan-300">Frames ${{nodeData.start_frame}} - ${{nodeData.end_frame}}</span></div>
                        <div><span class="text-slate-400">Duration:</span> <span class="font-mono">${{nodeData.duration}} frames</span></div>
                        <div><span class="text-slate-400">Parent Cell:</span> <span class="font-mono text-amber-300">${{nodeData.parent ? '#' + nodeData.parent : 'Root Ancestor'}}</span></div>
                        <div><span class="text-slate-400">Mitosis Event:</span> <span class="font-mono text-emerald-400">${{nodeData.has_division ? 'Yes (' + (nodeData.children ? nodeData.children.length : 0) + ' Daughters)' : 'No'}}</span></div>
                        <div><span class="text-slate-400">Mean Speed:</span> <span class="font-mono">${{nodeData.speed || 0}} px/fr</span></div>
                        <div><span class="text-slate-400">Directionality:</span> <span class="font-mono">${{nodeData.directionality || 0}}</span></div>
                    </div>
                </div>
            `;
        }}

        function inspectCellById(cId) {{
            const found = lineageData.cell_lifespans.find(c => c.id === cId);
            if(found) inspectCellNode({{
                id: found.id,
                start_frame: found.start,
                end_frame: found.end,
                duration: found.end - found.start + 1,
                parent: found.parent,
                has_division: found.children && found.children.length > 0,
                speed: found.speed,
                children: found.children
            }});
        }}

        // Initial Draw
        window.onload = () => renderActiveFamily();
    </script>
</body>
</html>
"""
    out_path = Path("lineage_tree_viewer.html")
    out_path.write_text(html_content, encoding="utf-8")
    print(f"[LineageVisualizer] Generated lineage HTML visualizer: {out_path.resolve()}")
    return out_path


if __name__ == "__main__":
    build_lineage_web_visualizer()
