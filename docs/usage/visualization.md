# `math_assistant_agent.visualization`

Renders a `graph_data` dict (from [`data.build_graph_records`](data.md), optionally enriched by
[`enrichment`](enrichment.md)) to an interactive HTML page using [PyVis](https://pyvis.readthedocs.io/).

## `visualization.pyvis_renderer.render_graph`

```python
def render_graph(graph_data, output_path="knowledge_graph.html")
```

Builds a directed PyVis `Network` (dark background, `forceAtlas2Based` physics so the graph auto-arranges
itself) and writes it to `output_path`. Each node shows its label + hover tooltip (title, score, view
count where available); each edge is labeled with its relationship `type` (e.g. `HAS_ACCEPTED_ANSWER`,
`TAGGED_WITH`, `INCLUDES_CONCEPT`).

Node colors are keyed by `label`:

| Label | Color |
|---|---|
| `Question` | light red (`#FF9999`) |
| `Answer` | light green (`#99FF99`) |
| anything else (`Tag`, `Domain`, `Concept`, `ResolutionStep`, ...) | light blue (`#97C2FC`, default) |

```python
from math_assistant_agent.data import fetch_math_dataset, build_graph_records
from math_assistant_agent.visualization import render_graph

raw_items = fetch_math_dataset(num_questions=20)
graph_data = build_graph_records(raw_items)

render_graph(graph_data, output_path="graph_math_kb.html")
# Rendering visualization for 52 nodes and 40 edges...
# ✅ Graph rendered successfully! Open 'graph_math_kb.html' in your browser.
```

Open the resulting file directly in a browser — no server needed. Works the same whether `graph_data` has
just been built by `build_graph_records` or has already been through
[`enrichment.enrich_graph_records`](enrichment.md) (Domain/Concept/ResolutionStep nodes render with the
default color, since they aren't in the `Question`/`Answer` table above).
