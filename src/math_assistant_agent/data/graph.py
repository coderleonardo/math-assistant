import json

from math_assistant_agent.data.cleaning import clean_html_for_math


def build_graph_records(dados):
    """
    Converte pares Pergunta/Resposta em nós e arestas para uma base de
    conhecimento em grafo: nós Question e Answer ligados por HAS_ACCEPTED_ANSWER,
    e nós Tag ligados às Questions por TAGGED_WITH.
    """
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Grafo de conhecimento salvo em: {path}")
    return path


def get_node_by_id(graph_data, node_id):
    """Returns the full node dict for the given id, or None if not found."""
    for node in graph_data["nodes"]:
        if node["id"] == node_id:
            return node
    return None


def get_accepted_answer(graph_data, question_id):
    """Returns the Answer node linked to question_id via HAS_ACCEPTED_ANSWER, or None."""
    for edge in graph_data["edges"]:
        if edge["source"] == question_id and edge["type"] == "HAS_ACCEPTED_ANSWER":
            return get_node_by_id(graph_data, edge["target"])
    return None


def get_questions_by_min_score(graph_data, min_score=100):
    """Returns all Question nodes with score strictly greater than min_score."""
    return [
        node
        for node in graph_data["nodes"]
        if node["label"] == "Question" and (node["properties"].get("score") or 0) > min_score
    ]
