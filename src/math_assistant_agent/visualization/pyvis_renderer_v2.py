from pyvis.network import Network

NODE_COLORS = {
    "Question": "#FF9999",  # light red
    "Answer": "#99FF99",    # light green
    "Tag": "#FFD480",       # light orange
    "Concept": "#C9A0FF",   # light purple
    "Theorem": "#99CCFF",   # light blue — NEW: for our formulas and theorems
}
DEFAULT_NODE_COLOR = "#D3D3D3"  # light grey

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
    """Render graph_data as an interactive PyVis HTML file at output_path."""
    print(
        f"Rendering visualization for {len(graph_data.get('nodes', []))} nodes "
        f"and {len(graph_data.get('edges', []))} edges..."
    )

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,
        notebook=False,  
    )

    # 1. Add nodes
    for node in graph_data.get("nodes", []):
        node_id = node["id"]
        label = node["label"]
        props = node.get("properties", {})

        # AJUSTE 1: Fallback para 'name' caso 'title' não exista (para Conceitos e Teoremas)
        title = props.get("title", props.get("name", "No title/name"))
        score = props.get("score", "N/A")
        views = props.get("view_count", "N/A")

        hover_text = f"<b>Label:</b> {label}<br><b>Name/Title:</b> {title}<br><b>Score:</b> {score} | <b>Views:</b> {views}"

        # AJUSTE 2: Exibir o LaTeX no hover se for um Teorema
        latex = props.get("latex")
        if latex and latex != "N/A":
            hover_text += f"<br><b>LaTeX:</b> {latex}"

        # Resolution steps and aliases (Mantido do seu original)
        steps = props.get("resolution_steps")
        if steps:
            hover_text += f"<br><b>Resolution steps:</b> {len(steps)}"

        aliases = props.get("aliases")
        if aliases and len(aliases) > 1:
            hover_text += f"<br><b>Also known as:</b> {', '.join(aliases)}"

        color = NODE_COLORS.get(label, DEFAULT_NODE_COLOR)

        # Usar o nome truncado para a label visível se for Conceito/Teorema ajuda na leitura
        # Se não, fallback para o sufixo do ID.
        visible_name = props.get("name", node_id.split('_')[-1])
        if len(visible_name) > 15:
            visible_name = visible_name[:15] + "..."
            
        display_label = f"{label}\n({visible_name})"

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
            title=edge["type"],  
            label=edge["type"],  
            color="#888888",
        )

    # 3. Physics
    net.set_options(PHYSICS_OPTIONS)

    # 4. Write
    net.write_html(output_path)
    print(f"✅ Graph rendered successfully! Open '{output_path}' in your browser.")