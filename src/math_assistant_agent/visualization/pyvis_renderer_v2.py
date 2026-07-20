from pyvis.network import Network

NODE_COLORS = {
    "CanonicalProblem": "#FFD700", # Gold — The core distilled mathematical case
    "Concept": "#C9A0FF",          # Light purple
    "Theorem": "#99CCFF",          # Light blue
    # Legacy colors kept for backward compatibility / debugging
    "Question": "#FF9999",  
    "Answer": "#99FF99",    
    "Tag": "#FFD480",       
}
DEFAULT_NODE_COLOR = "#D3D3D3"  # Light grey

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

def render_graph(graph_data, output_path="knowledge_graph_cbr.html"):
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

        # Base hover text
        hover_text = f"<b>Label:</b> {label}<br>"

        # Lógica de Hover e Label dependendo do tipo do nó
        if label == "CanonicalProblem":
            objective = props.get("objective", "No objective defined")
            steps = props.get("solution_steps", [])
            hover_text += f"<b>Objective:</b> {objective}<br>"
            hover_text += f"<b>Solution Steps:</b> {len(steps)}<br>"
            
            # Label visível será o ID numérico original
            visible_name = node_id.split('_')[-1]
            
        else:
            # Concepts, Theorems ou nós antigos
            name = props.get("name", props.get("title", "No name"))
            hover_text += f"<b>Name/Title:</b> {name}<br>"
            
            latex = props.get("latex")
            if latex and latex != "N/A":
                hover_text += f"<b>LaTeX:</b> {latex}<br>"
                
            # Truncamento de nomes longos para a interface não ficar poluída
            visible_name = props.get("name", node_id.split('_')[-1])
            if len(visible_name) > 15:
                visible_name = visible_name[:15] + "..."

        # Mantido para nós legados que ainda possuam score/views
        if "score" in props:
            score = props.get("score", "N/A")
            views = props.get("view_count", "N/A")
            hover_text += f"<b>Score:</b> {score} | <b>Views:</b> {views}<br>"

        color = NODE_COLORS.get(label, DEFAULT_NODE_COLOR)
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