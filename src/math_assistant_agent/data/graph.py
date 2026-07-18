import json

from math_assistant_agent.config import METADATA_TAGS
from math_assistant_agent.data.cleaning import clean_html_for_math


def build_graph_records(dados, exclude_tags=METADATA_TAGS):
    """Convert Question/Answer pairs into a graph_data dict.

    Builds Question and Answer nodes linked by a HAS_ACCEPTED_ANSWER edge, and Tag
    nodes (deduplicated by name) linked to their Questions by TAGGED_WITH.

    Tags in exclude_tags are skipped. This is a curation choice rather than a correctness
    fix: meta tags like "big-list" do connect questions, but not by mathematical topic,
    and they dominate the graph's hubs. Pass exclude_tags=() to keep every tag.

    Example:
        >>> raw_items = fetch_math_dataset(num_questions=5)
        >>> graph_data = build_graph_records(raw_items)
        >>> graph_data["nodes"][0]["label"]
        'Question'
    """
    exclude_tags = frozenset(exclude_tags or ())

    nodes_by_id = {}
    edges = []

    for item in dados:
        question_id = item["question_id"]
        answer_id = item.get("answer_id")

        pergunta_limpa = clean_html_for_math(item["prompt"])
        resposta_limpa = clean_html_for_math(item["completion"])

        question_node_id = f"question_{question_id}"
        answer_node_id = f"answer_{answer_id}" if answer_id is not None else f"answer_{question_id}"

        nodes_by_id[question_node_id] = {
            "id": question_node_id,
            "label": "Question",
            "properties": {
                "question_id": question_id,
                "title": item["title"],
                "text": pergunta_limpa,
                "score": item.get("question_score"),
                "view_count": item.get("view_count"),
                "link": item.get("link"),
            },
        }

        nodes_by_id[answer_node_id] = {
            "id": answer_node_id,
            "label": "Answer",
            "properties": {
                "answer_id": answer_id,
                "question_id": question_id,
                "text": resposta_limpa,
                "score": item.get("answer_score"),
            },
        }

        edges.append(
            {
                "source": question_node_id,
                "target": answer_node_id,
                "type": "HAS_ACCEPTED_ANSWER",
                "properties": {},
            }
        )

        # Nós Tag são compartilhados entre perguntas, então só criamos um por nome.
        for tag in item.get("tags", []):
            if tag in exclude_tags:
                continue

            tag_node_id = f"tag_{tag}"
            nodes_by_id.setdefault(
                tag_node_id,
                {"id": tag_node_id, "label": "Tag", "properties": {"name": tag}},
            )
            edges.append(
                {
                    "source": question_node_id,
                    "target": tag_node_id,
                    "type": "TAGGED_WITH",
                    "properties": {},
                }
            )

    return {"nodes": list(nodes_by_id.values()), "edges": edges}


def save_graph_json(graph_data, path="graph_math_kb.json"):
    """Write graph_data to path as JSON and return path.

    Example:
        >>> save_graph_json(graph_data, path="data/graph_math_kb.json")
        'data/graph_math_kb.json'
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Grafo de conhecimento salvo em: {path}")
    return path


def load_graph_json(path):
    """Load a graph_data dict previously written by save_graph_json.

    Example:
        >>> graph_data = load_graph_json("data/graph_math_2026-07-17-16.json")
        >>> len(graph_data["nodes"])
        339
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prune_node_label(graph_data, label):
    """Remove every node with the given label and every edge incident to those nodes.

    Mutates and returns graph_data. Useful for dropping a whole layer (e.g. Domain)
    without leaving dangling edges behind.

    Example:
        >>> prune_node_label(graph_data, "Domain")  # drops Domain nodes + INCLUDES_CONCEPT edges
    """
    drop_ids = {node["id"] for node in graph_data["nodes"] if node["label"] == label}
    graph_data["nodes"] = [node for node in graph_data["nodes"] if node["id"] not in drop_ids]
    graph_data["edges"] = [
        edge
        for edge in graph_data["edges"]
        if edge["source"] not in drop_ids and edge["target"] not in drop_ids
    ]
    return graph_data


def prune_tags(graph_data, exclude_tags=METADATA_TAGS):
    """Remove Tag nodes whose name is in exclude_tags, plus their incident edges.

    The counterpart to build_graph_records's exclude_tags, for a graph that was already
    built with every tag — cleans it in place instead of re-fetching from StackExchange.
    Mutates and returns graph_data.

    Example:
        >>> prune_tags(graph_data)  # drops big-list, soft-question, intuition, ...
    """
    exclude_tags = frozenset(exclude_tags or ())

    drop_ids = {
        node["id"]
        for node in graph_data["nodes"]
        if node["label"] == "Tag" and node["properties"]["name"] in exclude_tags
    }

    graph_data["nodes"] = [node for node in graph_data["nodes"] if node["id"] not in drop_ids]
    graph_data["edges"] = [
        edge
        for edge in graph_data["edges"]
        if edge["source"] not in drop_ids and edge["target"] not in drop_ids
    ]
    return graph_data


def get_node_by_id(graph_data, node_id):
    """Return the full node dict for node_id, or None if not found.

    Example:
        >>> get_node_by_id(graph_data, "question_182412")["label"]
        'Question'
    """
    for node in graph_data["nodes"]:
        if node["id"] == node_id:
            return node
    return None


def get_accepted_answer(graph_data, question_id):
    """Return the Answer node linked to question_id via HAS_ACCEPTED_ANSWER, or None."""
    for edge in graph_data["edges"]:
        if edge["source"] == question_id and edge["type"] == "HAS_ACCEPTED_ANSWER":
            return get_node_by_id(graph_data, edge["target"])
    return None


def get_questions_by_min_score(graph_data, min_score=100):
    """Return all Question nodes with score strictly greater than min_score."""
    return [
        node
        for node in graph_data["nodes"]
        if node["label"] == "Question" and (node["properties"].get("score") or 0) > min_score
    ]
