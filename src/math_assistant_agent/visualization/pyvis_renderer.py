from pyvis.network import Network

NODE_COLORS = {
    "Question": "#FF9999",  # light red
    "Answer": "#99FF99",  # light green
    "Tag": "#FFD480",  # light orange — the macro topic spine
    "Concept": "#C9A0FF",  # light purple — the shared mid-layer
}
DEFAULT_NODE_COLOR = "#97C2FC"  # light blue

PHYSICS_OPTIONS = """
var options = {
  "physics": {
    "forceAtlas2Based": {
      "gravitationalConstant": -50,
      "centralGravity": 0.01,
      "springLength": 100,
      "springConstant": 0.08
    },
    "maxVelocity": 50,
    "solver": "forceAtlas2Based",
    "timestep": 0.35,
    "stabilization": {"iterations": 150}
  }
}
"""


def render_graph(graph_data, output_path="knowledge_graph.html"):
    """Render graph_data as an interactive PyVis HTML file at output_path.

    Nodes are colored by label (see NODE_COLORS/DEFAULT_NODE_COLOR); edges are labeled
    with their relationship type. Open output_path directly in a browser, no server
    needed.

    Example:
        render_graph(graph_data, output_path="graph_math_kb.html")
    """
    print(
        f"Rendering visualization for {len(graph_data.get('nodes', []))} nodes "
        f"and {len(graph_data.get('edges', []))} edges..."
    )

    # directed=True matters so we can see edge direction (e.g. HAS_ACCEPTED_ANSWER)
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,
        notebook=False,  # switch to True to render directly in a Jupyter cell
    )

    # 1. Add nodes
    for node in graph_data.get("nodes", []):
        node_id = node["id"]
        label = node["label"]
        props = node.get("properties", {})

        title = props.get("title", "No title")
        score = props.get("score", "N/A")
        views = props.get("view_count", "N/A")

        hover_text = f"<b>Label:</b> {label}<br><b>Title:</b> {title}<br><b>Score:</b> {score} | <b>Views:</b> {views}"

        # Resolution steps ride on the Answer node rather than being nodes themselves,
        # so surface them on hover or they'd be invisible in the rendering.
        steps = props.get("resolution_steps")
        if steps:
            hover_text += f"<br><b>Resolution steps:</b> {len(steps)}"

        # Same for the surface forms a Concept absorbed during entity resolution.
        aliases = props.get("aliases")
        if aliases and len(aliases) > 1:
            hover_text += f"<br><b>Also known as:</b> {', '.join(aliases)}"

        color = NODE_COLORS.get(label, DEFAULT_NODE_COLOR)

        # visible label = node type + tail of the id
        display_label = f"{label}\n({node_id.split('_')[-1]})"

        net.add_node(
            n_id=node_id,
            label=display_label,
            title=hover_text,
            color=color,
        )

    # 2. Add edges
    for edge in graph_data.get("edges", []):
        net.add_edge(
            source=edge["source"],
            to=edge["target"],
            title=edge["type"],  # tooltip on hover
            label=edge["type"],  # visible label on the edge
            color="#888888",
        )

    # 3. Physics so the graph auto-arranges itself
    net.set_options(PHYSICS_OPTIONS)

    # 4. Write the HTML file
    net.write_html(output_path)
    print(f"✅ Graph rendered successfully! Open '{output_path}' in your browser.")
