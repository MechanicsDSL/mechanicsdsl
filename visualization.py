"""
MechanicsDSL Dependency Graph Visualizer
Generates two publication-quality graphs:
  1. Full graph colored by role/category
  2. Full graph colored by direct vs transitive dependency

Usage:
    pip install networkx matplotlib
    pipdeptree --json > deps.json
    python visualize_deps.py
"""

import json
import sys
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

DEPS_FILE    = "deps.json"
ROOT_PACKAGE = "mechanicsdsl-core"

CATEGORIES = {
    "MechanicsDSL": {
        "packages": ["mechanicsdsl", "mechanicsdsl-core"],
        "color": "#E63946",
    },
    "Parser / Compiler": {
        "packages": ["lark", "parso", "lark-parser"],
        "color": "#F4A261",
    },
    "Scientific Core": {
        "packages": [
            "numpy", "scipy", "sympy", "numba", "mpmath",
            "pandas", "tqdm",
        ],
        "color": "#2A9D8F",
    },
    "Visualization": {
        "packages": [
            "matplotlib", "plotly", "contourpy", "pillow",
            "matplotlib-inline", "colorama", "webcolors",
            "kiwisolver", "cycler",
        ],
        "color": "#457B9D",
    },
    "Packaging / Infra": {
        "packages": [
            "setuptools", "packaging", "pip", "certifi",
            "fonttools", "python-dateutil", "typing-extensions",
            "six", "pyparsing", "kiwisolver", "cycler",
        ],
        "color": "#CDB4DB",
    },
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def normalize(name: str) -> str:
    return name.lower().replace("_", "-")


def load_deps(path: str):
    with open(path) as f:
        return json.load(f)


def build_graph(data):
    G = nx.DiGraph()
    direct_deps = set()

    pkg_lookup = {normalize(p["package"]["package_name"]): p for p in data}
    root_node  = pkg_lookup.get(normalize(ROOT_PACKAGE))

    if root_node is None:
        print(f"[warn] Root '{ROOT_PACKAGE}' not found — building full env graph.")
        for pkg in data:
            pname = normalize(pkg["package"]["package_name"])
            G.add_node(pname)
            for dep in pkg.get("dependencies", []):
                dname = normalize(dep["package_name"])
                G.add_edge(pname, dname)
                direct_deps.add(dname)
        return G, direct_deps

    root_name = normalize(root_node["package"]["package_name"])
    G.add_node(root_name)

    queue   = [root_node]
    visited = set()

    while queue:
        node  = queue.pop(0)
        nname = normalize(node["package"]["package_name"])
        if nname in visited:
            continue
        visited.add(nname)

        for dep in node.get("dependencies", []):
            dname = normalize(dep["package_name"])
            G.add_edge(nname, dname)
            if nname == root_name:
                direct_deps.add(dname)
            if dname not in visited and dname in pkg_lookup:
                queue.append(pkg_lookup[dname])

    return G, direct_deps


def get_category(node: str):
    n = normalize(node)
    for cat, info in CATEGORIES.items():
        if n in [normalize(p) for p in info["packages"]]:
            return cat, info["color"]
    return "Other", "#888888"


def draw_graph(
    G,
    node_colors,
    node_sizes,
    title,
    filename,
    legend_handles,
    edge_alpha=0.35,
    figsize=(16, 12),
):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    print(f"  Computing layout for '{title}' ({len(G.nodes)} nodes)...")

    root_norm = normalize(ROOT_PACKAGE)

    # Step 1: kamada_kawai for overall structure
    pos = nx.kamada_kawai_layout(G, scale=2.5)

    # Step 2: spring pass to spread tight clusters, root pinned at its position
    fixed_nodes = [root_norm] if root_norm in pos else []
    pos = nx.spring_layout(
        G,
        pos=pos,
        fixed=fixed_nodes if fixed_nodes else None,
        k=1.8,
        iterations=60,
        seed=42,
    )

    nx.draw_networkx_edges(
        G, pos,
        edge_color="#AAAAAA",
        alpha=edge_alpha,
        arrows=True,
        arrowsize=12,
        arrowstyle="-|>",
        width=1.2,
        min_source_margin=18,
        min_target_margin=18,
        ax=ax,
    )

    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.95,
        linewidths=1.5,
        edgecolors="#FFFFFF",
        ax=ax,
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_color="#FFFFFF",
        font_family="monospace",
        font_weight="bold",
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="#0D1117",
            edgecolor="none",
            alpha=0.65,
        ),
        ax=ax,
    )

    ax.set_title(
        title,
        fontsize=20,
        fontweight="bold",
        color="white",
        pad=24,
        fontfamily="monospace",
    )

    ax.legend(
        handles=legend_handles,
        loc="lower left",
        framealpha=0.4,
        facecolor="#1C2128",
        edgecolor="#444",
        labelcolor="white",
        fontsize=10,
        borderpad=0.8,
    )

    ax.axis("off")
    plt.tight_layout(pad=1.5)
    plt.savefig(filename, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved -> {filename}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    if not Path(DEPS_FILE).exists():
        print(f"ERROR: '{DEPS_FILE}' not found.")
        print("Run:  pipdeptree --json > deps.json")
        sys.exit(1)

    print("Loading dependency data...")
    data = load_deps(DEPS_FILE)

    print("Building graph...")
    G, direct_deps = build_graph(data)
    nodes = list(G.nodes())
    print(f"  {len(nodes)} nodes, {len(G.edges())} edges")
    print(f"  {len(direct_deps)} direct  |  {len(nodes) - len(direct_deps) - 1} transitive")

    root_norm   = normalize(ROOT_PACKAGE)
    direct_norm = {normalize(d) for d in direct_deps}

    # ── Graph 1: By Role ──────────────────────────────────
    print("\nRendering Graph 1: Role / Category...")

    cat_colors = []
    cat_sizes  = []
    seen_cats  = set()

    for n in nodes:
        cat, color = get_category(n)
        seen_cats.add(cat)
        cat_colors.append(color)
        if normalize(n) == root_norm:
            cat_sizes.append(3200)
        elif cat == "MechanicsDSL":
            cat_sizes.append(2000)
        else:
            cat_sizes.append(1400)

    legend_cat = []
    for cat, info in CATEGORIES.items():
        if cat in seen_cats:
            legend_cat.append(mpatches.Patch(color=info["color"], label=cat))
    if "Other" in seen_cats:
        legend_cat.append(mpatches.Patch(color="#888888", label="Other"))

    draw_graph(
        G,
        node_colors=cat_colors,
        node_sizes=cat_sizes,
        title="MechanicsDSL — Dependencies by Role",
        filename="graph1_by_role.png",
        legend_handles=legend_cat,
        figsize=(16, 12),
    )

    # ── Graph 2: Direct vs Transitive ─────────────────────
    print("\nRendering Graph 2: Direct vs Transitive...")

    ROOT_COLOR       = "#E63946"
    DIRECT_COLOR     = "#58A6FF"
    TRANSITIVE_COLOR = "#8B949E"

    dt_colors = []
    dt_sizes  = []

    for n in nodes:
        nn = normalize(n)
        if nn == root_norm:
            dt_colors.append(ROOT_COLOR)
            dt_sizes.append(3200)
        elif nn in direct_norm:
            dt_colors.append(DIRECT_COLOR)
            dt_sizes.append(1800)
        else:
            dt_colors.append(TRANSITIVE_COLOR)
            dt_sizes.append(1000)

    n_transitive = len(nodes) - len(direct_deps) - 1
    legend_dt = [
        mpatches.Patch(color=ROOT_COLOR,       label=f"Root  ({ROOT_PACKAGE})"),
        mpatches.Patch(color=DIRECT_COLOR,     label=f"Direct  ({len(direct_deps)})"),
        mpatches.Patch(color=TRANSITIVE_COLOR, label=f"Transitive  ({n_transitive})"),
    ]

    draw_graph(
        G,
        node_colors=dt_colors,
        node_sizes=dt_sizes,
        title="MechanicsDSL — Direct vs Transitive Dependencies",
        filename="graph2_direct_vs_transitive.png",
        legend_handles=legend_dt,
        figsize=(16, 12),
    )

    print("\nDone!")
    print("  graph1_by_role.png")
    print("  graph2_direct_vs_transitive.png")


if __name__ == "__main__":
    main()