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
    """True if question_id's accepted Answer already carries resolution_steps."""
    answer_node = get_accepted_answer(graph_data, question_id)
    if answer_node is None:
        return False
    return bool(answer_node["properties"].get("resolution_steps"))


def enrich_graph_with_entities(graph_data, question_id, extracted):
    """Inject one question's extracted entities into graph_data.

    Concepts become shared nodes linked to the question by APPLIES_TO_PROBLEM. Resolution
    steps are *not* nodes: they are stored as an ordered list on the question's accepted
    Answer node, because a solution's steps are private to that solution and no traversal
    ever crosses between two questions' step chains. Nodes are reserved for things that
    can be shared.

    extracted is the dict returned by extract_graph_entities /
    extract_graph_entities_groq. Concept nodes are deduplicated by name, keeping the
    first description seen. Mutates and returns graph_data.
    """
    for concept in extracted["concepts"]:
        concept_id = f"concept_{slugify_id(concept['name'])}"

        if get_node_by_id(graph_data, concept_id) is None:
            graph_data["nodes"].append(
                {
                    "id": concept_id,
                    "label": "Concept",
                    "properties": {
                        "name": concept["name"],
                        "description": concept.get("description", ""),
                    },
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

    answer_node = get_accepted_answer(graph_data, question_id)
    if answer_node is not None:
        answer_node["properties"]["resolution_steps"] = sorted(
            extracted["resolution_steps"], key=lambda step: step["step_number"]
        )

    return graph_data


# Distinguishes "caller said nothing about checkpointing" from an explicit
# checkpoint_path=None, which means "do not write to disk". Without this, passing a source
# path and checkpoint_path=None silently overwrites the source file.
_UNSET = object()


def enrich_graph_records(
    graph_data,
    client=None,
    extract_fn=extract_graph_entities,
    sleep_seconds=0,
    checkpoint_path=_UNSET,
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
    frame. If graph_data was passed as a path string and checkpoint_path is *omitted*, it
    defaults to that same path — so the source file is enriched in place. Pass an explicit
    checkpoint_path to keep a pristine raw graph you can re-run from, or an explicit
    checkpoint_path=None to disable disk writes entirely. When nothing is written to disk
    the only result is the returned dict and a crash mid-run loses everything, so a
    UserWarning is emitted in that case.

    Concept names already in the graph are passed to extract_fn as known_concepts so the
    LLM reuses them instead of inventing new phrasings for the same idea (controlled
    vocabulary). The list is recomputed each iteration, so a concept introduced by an
    earlier question in the run is reusable by later ones. A custom extract_fn must
    therefore accept a known_concepts keyword (both built-in extractors do). This only
    reduces fragmentation at the margin — run enrichment.resolve_concepts afterwards for
    the pass that actually consolidates the Concept layer.

    Note that when graph_data is a path, checkpoint_path defaults to that same path, so
    the source file is overwritten in place. Pass an explicit checkpoint_path to keep a
    pristine raw graph to re-run from.

    Example:
        >>> graph_data = enrich_graph_records(
        ...     "data/graph_math_2026-07-17-16.json",       # raw input, left untouched
        ...     checkpoint_path="data/graph_math_kb.json",  # enriched output
        ... )
    """
    if isinstance(graph_data, str):
        if checkpoint_path is _UNSET:
            checkpoint_path = graph_data
        graph_data = load_graph_json(graph_data)

    if checkpoint_path is _UNSET:
        checkpoint_path = None

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
            print(f"Extracted concepts: {[c['name'] for c in extracted['concepts']]}")

            if checkpoint_path and checkpoint_every and enriched_count % checkpoint_every == 0:
                save_graph_json(graph_data, path=checkpoint_path)

        if sleep_seconds and index < len(question_nodes) - 1:
            time.sleep(sleep_seconds)

    if checkpoint_path:
        save_graph_json(graph_data, path=checkpoint_path)

    return graph_data
