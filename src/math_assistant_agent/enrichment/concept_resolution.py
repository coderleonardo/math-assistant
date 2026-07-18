"""Consolidate the Concept layer by clustering surface-form variants into canonical nodes.

Extraction runs one question at a time, so the same mathematical idea comes back phrased
differently each time ("Kunneth Formula" / "Künneth theorem" / "Künneth's short exact
sequence"). Asking the extractor to reuse existing names (known_concepts) helps only at
the margin, because the duplication is semantic rather than orthographic — string
normalization merges essentially nothing.

The fix is a separate global pass over every extracted concept at once, using each
concept's one-sentence description as disambiguation context. That is what this module
does: resolve_concepts asks an LLM to cluster the names, and apply_concept_resolution
rewrites the graph so each cluster becomes a single node.
"""

import json

import groq
from google.genai import types
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from math_assistant_agent.config import (
    GEMINI_EXTRACTION_TEMPERATURE,
    GEMINI_MODEL_NAME,
    GROQ_EXTRACTION_TEMPERATURE,
    GROQ_MODEL_NAME,
)
from math_assistant_agent.enrichment.schemas import ResolvedConcepts

RESOLUTION_INSTRUCTIONS = """
You are a mathematician curating the vocabulary of a knowledge graph.
Below are mathematical concept names extracted from several different problems. Some are
different surface forms of the same underlying concept.

Cluster them. Rules:
- Every input name must appear in exactly one cluster's aliases list.
- A concept that is genuinely distinct forms its own single-element cluster.
- Use the descriptions to avoid merging concepts that merely share a name or a keyword.
- Do NOT merge a specific result into a broader area (e.g. "Cayley-Hamilton Theorem" must
  not be folded into "Linear Algebra"): only merge names that denote the same thing.
- The canonical name should be the most standard, unambiguous mathematical form.
"""


def _format_concepts(concepts):
    """Render concepts as the '- name: description' block used in the resolution prompt."""
    return "\n".join(f"- {c['name']}: {c.get('description', '')}".rstrip(": ") for c in concepts)


def resolve_concepts_gemini(client, concepts):
    """Cluster concepts into canonical/alias groups using Gemini structured output.

    concepts is a list of {"name", "description"} dicts. Returns a list of cluster dicts
    ({"canonical", "aliases"}), or [] on error so a multi-block run can continue.

    Example:
        client = build_gemini_client()
        clusters = resolve_concepts_gemini(client, [{"name": "Kunneth Formula", ...}])
    """
    prompt = f"{RESOLUTION_INSTRUCTIONS}\n<concepts>\n{_format_concepts(concepts)}\n</concepts>\n"

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ResolvedConcepts,
                temperature=GEMINI_EXTRACTION_TEMPERATURE,
            ),
        )

        return json.loads(response.text)["clusters"]

    except Exception as e:
        print(f"Error resolving concepts with Gemini: {e}")
        return []


@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(groq.RateLimitError),
    reraise=True,
)
def resolve_concepts_groq(client, concepts):
    """Cluster concepts into canonical/alias groups using Groq's JSON mode.

    Same signature as resolve_concepts_gemini — pass either as resolve_concepts's
    resolve_fn. Retries up to 5 times with exponential backoff on groq.RateLimitError;
    any other error is printed and returns [].

    Example:
        client = build_groq_client()
        clusters = resolve_concepts_groq(client, [{"name": "Kunneth Formula", ...}])
    """
    json_schema = ResolvedConcepts.model_json_schema()

    system_prompt = f"""{RESOLUTION_INSTRUCTIONS}
    You MUST respond ONLY with a valid JSON object that strictly follows this schema:
    {json.dumps(json_schema, indent=2)}
    """

    user_prompt = f"<concepts>\n{_format_concepts(concepts)}\n</concepts>"

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=GROQ_EXTRACTION_TEMPERATURE,
        )

        data = json.loads(response.choices[0].message.content)
        validated = ResolvedConcepts(**data)

        return [cluster.model_dump() for cluster in validated.clusters]

    except groq.RateLimitError:
        raise
    except Exception as e:
        print(f"Error resolving concepts with Groq: {e}")
        return []


def build_alias_map(clusters, all_names):
    """Flatten clusters into an {alias: canonical} dict covering every name in all_names.

    Any name the model left out of every cluster maps to itself, so a concept can never
    silently vanish from the graph just because the resolver forgot to mention it.

    Example:
        >>> build_alias_map([{"canonical": "Kunneth Formula",
        ...                   "aliases": ["Kunneth Formula", "Kunneth theorem"]}],
        ...                 ["Kunneth Formula", "Kunneth theorem", "Tensor Product"])
        {'Kunneth Formula': 'Kunneth Formula', 'Kunneth theorem': 'Kunneth Formula', 'Tensor Product': 'Tensor Product'}
    """
    alias_map = {}

    for cluster in clusters:
        canonical = cluster["canonical"]
        for alias in cluster["aliases"]:
            alias_map[alias] = canonical

    for name in all_names:
        alias_map.setdefault(name, name)

    return alias_map


def _concept_descriptions(graph_data):
    """Collect Concept nodes as {"name", "description"} dicts.

    Concepts extracted before the description field existed fall back to the titles of
    the questions they apply to, so resolution still has disambiguation context.
    """
    question_titles = {
        node["id"]: node["properties"].get("title", "")
        for node in graph_data["nodes"]
        if node["label"] == "Question"
    }

    linked_titles = {}
    for edge in graph_data["edges"]:
        if edge["type"] == "APPLIES_TO_PROBLEM":
            title = question_titles.get(edge["target"])
            if title:
                linked_titles.setdefault(edge["source"], []).append(title)

    concepts = []
    for node in graph_data["nodes"]:
        if node["label"] != "Concept":
            continue

        description = node["properties"].get("description", "")
        if not description:
            titles = linked_titles.get(node["id"], [])
            description = f"Used in: {'; '.join(titles[:3])}" if titles else ""

        concepts.append({"name": node["properties"]["name"], "description": description})

    return concepts


def resolve_concepts(
    graph_data,
    client=None,
    resolve_fn=resolve_concepts_gemini,
    block_size=80,
):
    """Cluster every Concept in graph_data and return an {alias: canonical} map.

    Concepts are resolved in blocks of block_size rather than one giant prompt, then the
    resulting canonical names are resolved once more so duplicates that landed in
    different blocks still merge. A few hundred concepts costs roughly
    len(concepts)/block_size + 1 calls.

    Pass the map to apply_concept_resolution to actually rewrite the graph; keeping the
    two steps separate means the map can be inspected (and edited) before anything is
    changed, and re-applied without spending tokens again.

    resolve_fn defaults to the Gemini backend; pass resolve_fn=resolve_concepts_groq
    together with client=build_groq_client() to use Groq instead. When client is None a
    Gemini client is built automatically, so pass an explicit client for any other
    resolve_fn.

    Example:
        alias_map = resolve_concepts(graph_data)
        apply_concept_resolution(graph_data, alias_map)
    """
    # Imported here to avoid a circular import at module load.
    from math_assistant_agent.enrichment.gemini_client import build_gemini_client

    if client is None:
        client = build_gemini_client()

    concepts = _concept_descriptions(graph_data)
    if not concepts:
        return {}

    all_names = [c["name"] for c in concepts]

    def resolve_in_blocks(items, label):
        clusters = []
        for start in range(0, len(items), block_size):
            block = items[start : start + block_size]
            print(f"{label}: resolving {start + 1}-{start + len(block)} of {len(items)}...")
            clusters.extend(resolve_fn(client, block))
        return clusters

    first_pass = build_alias_map(resolve_in_blocks(concepts, "Pass 1"), all_names)

    # Second round over the canonicals only, so two variants that landed in different
    # blocks still meet. Descriptions carry over from the concept each canonical came from.
    description_by_name = {c["name"]: c["description"] for c in concepts}
    canonicals = sorted(set(first_pass.values()))
    second_input = [
        {"name": name, "description": description_by_name.get(name, "")} for name in canonicals
    ]
    second_pass = build_alias_map(resolve_in_blocks(second_input, "Pass 2"), canonicals)

    alias_map = {name: second_pass[canonical] for name, canonical in first_pass.items()}

    resolved_count = len(set(alias_map.values()))
    print(
        f"Resolved {len(alias_map)} concept names into {resolved_count} concepts "
        f"({len(alias_map) - resolved_count} merged)."
    )

    return alias_map


def apply_concept_resolution(graph_data, alias_map):
    """Rewrite graph_data so every Concept is replaced by its canonical form.

    Merged nodes keep the longest available description and record the surface forms they
    absorbed in an "aliases" property. APPLIES_TO_PROBLEM edges are repointed at the
    canonical node and deduplicated — without that, two concepts that merge into one while
    both pointing at the same question would leave a parallel edge behind.

    Takes no LLM calls: pass it the map from resolve_concepts. Mutates and returns
    graph_data.

    Example:
        apply_concept_resolution(graph_data, {"Kunneth theorem": "Kunneth Formula"})
    """
    from math_assistant_agent.enrichment.graph_enrichment import slugify_id

    id_remap = {}
    merged_nodes = {}
    other_nodes = []

    for node in graph_data["nodes"]:
        if node["label"] != "Concept":
            other_nodes.append(node)
            continue

        name = node["properties"]["name"]
        canonical = alias_map.get(name, name)
        canonical_id = f"concept_{slugify_id(canonical)}"
        id_remap[node["id"]] = canonical_id

        existing = merged_nodes.get(canonical_id)
        if existing is None:
            merged_nodes[canonical_id] = {
                "id": canonical_id,
                "label": "Concept",
                "properties": {
                    "name": canonical,
                    "description": node["properties"].get("description", ""),
                    "aliases": sorted({name, canonical}),
                },
            }
            continue

        # Keep the most informative description and remember every surface form merged in.
        description = node["properties"].get("description", "")
        if len(description) > len(existing["properties"]["description"]):
            existing["properties"]["description"] = description
        existing["properties"]["aliases"] = sorted(set(existing["properties"]["aliases"]) | {name})

    graph_data["nodes"] = other_nodes + list(merged_nodes.values())

    seen_edges = set()
    rewritten_edges = []
    for edge in graph_data["edges"]:
        source = id_remap.get(edge["source"], edge["source"])
        target = id_remap.get(edge["target"], edge["target"])

        key = (source, target, edge["type"])
        if key in seen_edges:
            continue
        seen_edges.add(key)

        rewritten_edges.append({**edge, "source": source, "target": target})

    graph_data["edges"] = rewritten_edges

    return graph_data
