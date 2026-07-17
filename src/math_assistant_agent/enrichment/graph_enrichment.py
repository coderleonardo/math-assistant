import re

from math_assistant_agent.data.graph import get_accepted_answer, get_node_by_id
from math_assistant_agent.enrichment.extractor import extract_graph_entities
from math_assistant_agent.enrichment.gemini_client import build_gemini_client


def slugify_id(text):
    """Turns a concept/domain name into a valid node id (lowercase, no spaces)."""
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", text)


def enrich_graph_with_entities(graph_data, question_id, extracted):
    """
    Reads the structured JSON extracted by Gemini and injects nodes/edges into
    graph_data in place, building the Domain -> Concept -> Question -> ResolutionStep
    hierarchy (Graph Chain of Thought). Mutates and returns graph_data.
    """
    domain_id = f"domain_{slugify_id(extracted['domain'])}"
    if get_node_by_id(graph_data, domain_id) is None:
        graph_data["nodes"].append(
            {
                "id": domain_id,
                "label": "Domain",
                "properties": {"name": extracted["domain"]},
            }
        )

    for concept in extracted["concepts"]:
        concept_id = f"concept_{slugify_id(concept)}"

        if get_node_by_id(graph_data, concept_id) is None:
            graph_data["nodes"].append(
                {
                    "id": concept_id,
                    "label": "Concept",
                    "properties": {"name": concept},
                }
            )

            graph_data["edges"].append(
                {
                    "source": domain_id,
                    "target": concept_id,
                    "type": "INCLUDES_CONCEPT",
                    "properties": {},
                }
            )

        graph_data["edges"].append(
            {
                "source": concept_id,
                "target": question_id,
                "type": "APPLIES_TO_PROBLEM",
                "properties": {},
            }
        )

    previous_step_id = question_id
    link_type = "HAS_FIRST_STEP"

    for step in extracted["resolution_steps"]:
        step_id = f"step_{question_id}_{step['step_number']}"

        graph_data["nodes"].append(
            {
                "id": step_id,
                "label": "ResolutionStep",
                "properties": {
                    "step_number": step["step_number"],
                    "description": step["description"],
                    "math_formula": step["formula_latex"],
                },
            }
        )

        graph_data["edges"].append(
            {
                "source": previous_step_id,
                "target": step_id,
                "type": link_type,
                "properties": {},
            }
        )

        previous_step_id = step_id
        link_type = "NEXT_STEP"

    return graph_data


def enrich_graph_records(graph_data, dados, client=None):
    """
    Batch convenience: for each raw item in dados, looks up its already-cleaned
    Question/Answer text in graph_data, calls Gemini to extract the graph entities,
    and applies enrich_graph_with_entities. Mutates and returns graph_data.
    """
    if client is None:
        client = build_gemini_client()

    for item in dados:
        question_id = f"question_{item['question_id']}"
        question_node = get_node_by_id(graph_data, question_id)
        if question_node is None:
            continue

        answer_node = get_accepted_answer(graph_data, question_id)
        answer_text = answer_node["properties"]["text"] if answer_node else ""

        extracted = extract_graph_entities(
            client,
            item["title"],
            question_node["properties"]["text"],
            answer_text,
        )

        if extracted:
            enrich_graph_with_entities(graph_data, question_id, extracted)
            print(f"Extracted: {extracted['domain']} | {extracted['concepts']}")

    return graph_data
