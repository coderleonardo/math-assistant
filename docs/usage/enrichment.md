# `math_assistant_agent.enrichment`

Adds a semantic layer on top of the Question/Answer/Tag graph from [`data.md`](data.md): reads each
question + accepted answer with the Gemini API (structured JSON output via a Pydantic schema) and
extracts `Domain -> Concept -> Question -> ResolutionStep`, a "Graph Chain of Thought" — so a question no
longer points at one wall of text, but at a sequence of individually addressable, LaTeX-tagged reasoning
steps.

This module **adds nodes/edges to an existing `graph_data` dict** (from `data.build_graph_records`); it
never creates Question/Answer/Tag nodes itself.

Requires a `GEMINI_API_KEY` (get one from Google AI Studio) — see `.env.example` at the repo root.

## `enrichment.schemas`

```python
class ResolutionStep(BaseModel):
    step_number: int
    description: str
    formula_latex: str

class GraphExtraction(BaseModel):
    domain: str
    concepts: list[str]
    resolution_steps: list[ResolutionStep]
```

Pydantic models passed to Gemini as `response_schema` — the model is forced to return JSON matching this
shape exactly, no free-form text. `concepts` is capped at "up to 5" by the field description (a prompt
instruction, not an enforced constraint).

## `enrichment.gemini_client.build_gemini_client`

```python
def build_gemini_client(api_key=None)
```

Returns a `google.genai.Client`. If `api_key` isn't passed, falls back to the `GEMINI_API_KEY`
environment variable — same `api_key=None` fallback pattern as `data.fetch_math_dataset`. Raises
`ValueError` if neither is set (the one function in this module that raises instead of printing and
returning `None` — there's nothing useful to continue with without a client).

```python
from math_assistant_agent.enrichment import build_gemini_client

client = build_gemini_client()
# client = build_gemini_client(api_key="AI...")  # or pass explicitly
```

## `enrichment.extractor.extract_graph_entities`

```python
def extract_graph_entities(client, question_title, question_text, answer_text)
```

Calls `gemini-2.5-flash` (`config.GEMINI_MODEL_NAME`) with `temperature=0.1`
(`config.GEMINI_EXTRACTION_TEMPERATURE`) and `response_schema=GraphExtraction`, then parses the returned
JSON into a plain dict. On any error, prints it and returns `None` — matches `fetch_math_dataset`'s
no-raise convention, so a batch loop can skip a failed item instead of dying.

- `question_title`, `question_text`, `answer_text` — plain text, not HTML. Pass the already-cleaned
  `text` property from a Question/Answer node (see `enrich_graph_records` below), not raw
  `fetch_math_dataset` HTML.

```python
from math_assistant_agent.enrichment import build_gemini_client, extract_graph_entities

client = build_gemini_client()
extracted = extract_graph_entities(
    client,
    "Why do roots of polynomials tend to have absolute value close to 1?",
    "While playing around with Mathematica I noticed that most polynomials...",
    "This is a consequence of the distribution of roots of random polynomials...",
)
extracted
# {
#     "domain": "Complex Analysis",
#     "concepts": ["Residue Theorem", "Random Polynomials"],
#     "resolution_steps": [
#         {"step_number": 1, "description": "...", "formula_latex": "..."},
#         ...
#     ],
# }
```

## `enrichment.graph_enrichment.enrich_graph_with_entities`

```python
def enrich_graph_with_entities(graph_data, question_id, extracted)
```

Takes the dict returned by `extract_graph_entities` and injects nodes/edges into `graph_data`:

- One `Domain` node (`domain_<slug>`), deduplicated by name.
- One `Concept` node per concept (`concept_<slug>`), deduplicated by name, linked from `Domain` via
  `INCLUDES_CONCEPT` and to the `Question` via `APPLIES_TO_PROBLEM`.
- One `ResolutionStep` node per step (`step_<question_id>_<step_number>`), chained
  `Question -[HAS_FIRST_STEP]-> Step 1 -[NEXT_STEP]-> Step 2 -[NEXT_STEP]-> ...`.

**Mutates `graph_data` in place and also returns it** — unlike `build_graph_records`, which returns a
fresh dict. This is deliberate: Gemini calls are slow and can fail mid-batch, so a caller enriching
hundreds of questions will want to checkpoint with `save_graph_json` periodically rather than rebuild the
whole graph from scratch on every call. Pass a copy (`copy.deepcopy(graph_data)`) if you need to preserve
the pre-enrichment graph.

Domain/Concept dedup is checked via `data.graph.get_node_by_id`, so calling this repeatedly across many
questions that share a domain (e.g. "Calculus") only ever creates one `Domain` node.

```python
from math_assistant_agent.enrichment import enrich_graph_with_entities

enrich_graph_with_entities(graph_data, "question_182412", extracted)
```

## `enrichment.graph_enrichment.enrich_graph_records`

```python
def enrich_graph_records(graph_data, dados, client=None)
```

Batch convenience wrapper: for each raw item in `dados` (the list from `fetch_math_dataset`), looks up
its already-cleaned Question/Answer text in `graph_data` (via `get_node_by_id`/`get_accepted_answer` —
no re-cleaning HTML), calls `extract_graph_entities`, and applies `enrich_graph_with_entities`. Builds a
client with `build_gemini_client()` if none is passed. Skips items whose Question node isn't found in
`graph_data` (e.g. if you only ran `build_graph_records` on a subset).

```python
from math_assistant_agent.data import fetch_math_dataset, build_graph_records, save_graph_json
from math_assistant_agent.enrichment import enrich_graph_records

raw_items = fetch_math_dataset(api_key="YOUR_KEY", num_questions=50)
graph_data = build_graph_records(raw_items)

enrich_graph_records(graph_data, raw_items)
# Extracted: Complex Analysis | ['Residue Theorem', 'Random Polynomials']
# Extracted: Number Theory | ['ABC Conjecture', ...]
# ...

save_graph_json(graph_data, path="graph_math_kb.json")
```

For a large `raw_items` list, prefer chunking manually and calling `save_graph_json` between chunks, so a
crash or quota error partway through doesn't lose earlier progress:

```python
CHUNK = 20
for i in range(0, len(raw_items), CHUNK):
    enrich_graph_records(graph_data, raw_items[i : i + CHUNK])
    save_graph_json(graph_data, path="graph_math_kb.json")
```

Continue with [`visualization.md`](visualization.md) to render the enriched graph.
