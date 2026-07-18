import re
import time
import warnings

from math_assistant_agent.data.graph import (
    get_accepted_answer,
    get_node_by_id,
    load_graph_json,
    save_graph_json,
)
from math_assistant_agent.enrichment.extractor_gemini import extract_graph_entities
from math_assistant_agent.enrichment.gemini_client import build_gemini_client


def slugify_id(text):
    """Turn a concept name into a valid node id.

    Example:
        >>> slugify_id("Cayley-Hamilton Theorem")
        'cayley_hamilton_theorem'
    """
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", text)


def _is_already_enriched(graph_data, question_id):
    """True if question_id already has a HAS_FIRST_STEP edge (enrich_graph_with_entities ran for it)."""
    return any(
        edge["source"] == question_id and edge["type"] == "HAS_FIRST_STEP"
        for edge in graph_data["edges"]
    )


def enrich_graph_with_entities(graph_data, question_id, extracted):
    """Inject one question's extracted entities into graph_data as nodes/edges.

    Builds the Concept -> Question -> ResolutionStep hierarchy (Graph Chain of Thought)
    from extracted (the dict returned by extract_graph_entities /
    extract_graph_entities_groq). Concept nodes are deduplicated by name; ResolutionStep
    nodes are not deduplicated, so call this at most once per question_id
    (enrich_graph_records enforces that). Mutates and returns graph_data.
    """
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


def enrich_graph_records(
    graph_data,
    client=None,
    extract_fn=extract_graph_entities,
    sleep_seconds=0,
    checkpoint_path=None,
    checkpoint_every=10,
):
    """Enrich every not-yet-enriched Question node in graph_data with extracted entities.

    graph_data may be a path to a JSON file written by save_graph_json (loaded via
    load_graph_json) or an already-built graph_data dict, e.g. straight from
    build_graph_records. Questions that already have resolution steps attached are
    skipped, so it's safe to call this again on the same graph to resume after a crash
    or a quota error.

    extract_fn defaults to the Gemini backend (extract_graph_entities); pass
    extract_fn=extract_graph_entities_groq together with client=build_groq_client() to use
    Groq's gpt-oss models instead. When client is None, a Gemini client is built
    automatically — only the Gemini extract_fn has this default, so pass an explicit
    client for any other extract_fn.

    sleep_seconds paces requests between questions (skipped after the last one) to
    respect a provider's rate limits, e.g. sleep_seconds=15 for Groq's free-tier TPM
    limit. Defaults to 0 (no pause).

    checkpoint_path controls crash safety: if extract_fn raises (e.g. a rate limit
    retry finally exhausted), graph_data is saved to checkpoint_path *before*
    re-raising, and it's also saved every checkpoint_every successfully-enriched
    questions along the way — so a fatal error never strands progress in a dead stack
    frame. If graph_data was passed as a path string, checkpoint_path defaults to that
    same path automatically; if graph_data was passed as an in-memory dict,
    checkpoint_path defaults to None. In that None case nothing is written to disk (the
    only result is the returned dict, and a crash mid-run loses everything), so a
    UserWarning is emitted — pass checkpoint_path=... to enable iterative saving, or
    ignore the warning if you intend to save the returned value yourself.

    Concept names already in the graph are passed to extract_fn as known_concepts so the
    LLM reuses them instead of inventing new phrasings for the same idea (controlled
    vocabulary — this is what keeps the Concept layer connected rather than near-1:1 with
    questions). The list is recomputed each iteration, so a concept introduced by an
    earlier question in the run is reusable by later ones. A custom extract_fn must
    therefore accept a known_concepts keyword (both built-in extractors do).

    Example:
        >>> graph_data = enrich_graph_records("data/graph_math_2026-07-17-16.json")
        >>> save_graph_json(graph_data, path="data/graph_math_2026-07-17-16.json")
    """
    if isinstance(graph_data, str):
        if checkpoint_path is None:
            checkpoint_path = graph_data
        graph_data = load_graph_json(graph_data)

    if checkpoint_path is None:
        warnings.warn(
            "enrich_graph_records was called with an in-memory graph_data dict and no "
            "checkpoint_path, so nothing will be saved to disk: progress lives only in the "
            "returned value and a crash mid-run loses everything. Pass checkpoint_path=... "
            "to enable iterative saving, or save the returned graph_data yourself with "
            "save_graph_json.",
            stacklevel=2,
        )

    if client is None:
        client = build_gemini_client()

    question_nodes = [n for n in graph_data["nodes"] if n["label"] == "Question"]
    enriched_count = 0

    for index, question_node in enumerate(question_nodes):
        question_id = question_node["id"]
        if _is_already_enriched(graph_data, question_id):
            continue

        answer_node = get_accepted_answer(graph_data, question_id)
        answer_text = answer_node["properties"]["text"] if answer_node else ""

        # Feed back the concept names already in the graph so the LLM reuses them. Fine
        # to dump wholesale at this scale (hundreds); cap or embedding-retrieve for much
        # larger graphs.
        known_concepts = [
            node["properties"]["name"] for node in graph_data["nodes"] if node["label"] == "Concept"
        ]

        try:
            extracted = extract_fn(
                client,
                question_node["properties"]["title"],
                question_node["properties"]["text"],
                answer_text,
                known_concepts=known_concepts,
            )
        except Exception:
            if checkpoint_path:
                save_graph_json(graph_data, path=checkpoint_path)
            raise

        if extracted:
            enrich_graph_with_entities(graph_data, question_id, extracted)
            enriched_count += 1
            print(f"Extracted concepts: {extracted['concepts']}")

            if checkpoint_path and checkpoint_every and enriched_count % checkpoint_every == 0:
                save_graph_json(graph_data, path=checkpoint_path)

        if sleep_seconds and index < len(question_nodes) - 1:
            time.sleep(sleep_seconds)

    if checkpoint_path:
        save_graph_json(graph_data, path=checkpoint_path)

    return graph_data
