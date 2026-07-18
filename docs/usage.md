# Usage Guide

A narrative map of `src/math_assistant_agent/`, in pipeline order. Exhaustive per-function detail
(parameters, examples) lives in each function's docstring — this doc covers what each module is for and
the handful of things that aren't obvious from a one-line docstring alone.

## Pipeline overview

```python
from math_assistant_agent.data import (
    fetch_math_dataset, prepare_dataset, save_dataset_jsonl,
    build_graph_records, save_graph_json,
)
from math_assistant_agent.enrichment import (
    enrich_graph_records, resolve_concepts, apply_concept_resolution,
)
from math_assistant_agent.visualization import render_graph

raw_items = fetch_math_dataset(num_questions=50)

# Fine-tuning branch
save_dataset_jsonl(prepare_dataset(raw_items), path="dataset_math_qlora.jsonl")

# Knowledge-graph branch
graph_data = build_graph_records(raw_items)          # Question/Answer/Tag skeleton
save_graph_json(graph_data, path="data/graph_math_raw.json")

enrich_graph_records(                                 # + Concept nodes, + steps on Answers
    graph_data, checkpoint_path="data/graph_math_kb.json"
)
apply_concept_resolution(graph_data, resolve_concepts(graph_data))  # merge concept variants
save_graph_json(graph_data, path="data/graph_math_kb.json")
render_graph(graph_data, output_path="graph_math_kb.html")
```

## `config`

Dependency-free shared constants (`MODEL_NAME`, `THINK_END_TOKEN_ID`, `GENERATION_DEFAULTS`,
`GEMINI_*`/`GROQ_*` model+temperature pairs). One non-obvious bit: `SYSTEM_PROMPT` and
`AGENT_SYSTEM_PROMPT` are deliberately different strings — `SYSTEM_PROMPT` is what the model was actually
fine-tuned on (`data.formatting.prepare_dataset`, `inference.solver.solve`); `AGENT_SYSTEM_PROMPT` adds
Qwen3's required `\boxed{}` instruction on top, for the LangGraph agent path
(`agent.agent.build_math_agent`). Update `SYSTEM_PROMPT` here, not in the modules that import it.

## `data`

- `stackexchange.fetch_math_dataset` — pulls accepted-answer Q&A pairs from the MathOverflow API.
- `cleaning.clean_html_for_math` — strips HTML while preserving MathJax/LaTeX (`$...$`); this is the one
  invariant a generic HTML-stripping change could easily break.
- `formatting.prepare_dataset` / `save_dataset_jsonl` — the fine-tuning branch: raw items → ShareGPT-style
  JSONL for `SFTTrainer`.
- `graph.build_graph_records` / `save_graph_json` / `load_graph_json` — the knowledge-graph branch: raw
  items → a backend-agnostic `{"nodes": [...], "edges": [...]}` dict, and back from a JSON checkpoint.
- `graph.get_node_by_id` / `get_accepted_answer` / `get_questions_by_min_score` — read-only query helpers
  over a `graph_data` dict.
- `graph.prune_node_label` / `prune_tags` — drop a whole layer, or just the meta tags, without leaving
  dangling edges behind.

**Tag curation.** `build_graph_records(dados, exclude_tags=METADATA_TAGS)` skips StackExchange tags that
describe a question's *form* rather than its mathematics (`big-list`, `soft-question`, `intuition`, …;
the list lives in `config.METADATA_TAGS`). They were the graph's largest hubs by degree — `big-list` at
24 vs `nt.number-theory` at 13 — so the topic structure was routed through noise. Filtering them leaves a
purely mathematical spine. This is a curation choice, not a correctness fix: pass `exclude_tags=()` to
keep every tag, or use `prune_tags` to clean a graph that was already built with all of them.

It is a deliberate blocklist rather than a heuristic, because MathOverflow's `xx.` prefix convention can't
be used to infer it (`ho.history-overview` is a legitimate topic tag).

See `docs/_01_phases.md` for how this connects to Neo4j down the line (not wired up yet).

## `training`

QLoRA fine-tuning: `quantization.build_bnb_config`, `lora.build_lora_config`,
`model_loading.load_tokenizer` / `load_base_model_for_training`, and
`trainer.load_training_dataset` / `build_training_args` / `build_trainer` / `train_and_save_adapter`.

Two gotchas worth knowing before running this:
- `load_base_model_for_training` hardcodes `device_map={"": 0}` — `SFTTrainer` conflicts with
  `DataParallel` on multi-GPU machines, so training is pinned to a single device. Inference uses
  `device_map="auto"` instead (see below).
- `build_training_args`'s defaults include `max_steps=1` — that's a smoke-test value, not a real training
  config. Pass `num_train_epochs=...` as an override for an actual run.
- `train_and_save_adapter` saves only the LoRA adapter (a few MB), not the merged model.

## `inference`

Raw `transformers.generate()` path (no serving layer) — `model_loading.load_finetuned_model` attaches the
trained adapter to the 4-bit base model *without* merging it (`merge_and_unload` is not called here); the
tokenizer is loaded from `adapter_path`, not `base_model_name`, since that's where training saved it
alongside the weights.

`solver.solve(pergunta, model, tokenizer, **generation_overrides)` runs one turn and returns
`(thinking_content, final_answer)`, split by locating the `</think>` token id in the generated ids (not by
string-matching decoded text). **If you persist this into multi-turn history, only `final_answer` should
be fed back** — raw `thinking_content` in history degrades reasoning quality.

## `agent`

LangGraph agent-serving layer (Phase 4, in progress) — targets a locally-hosted OpenAI-compatible endpoint
(vLLM/Ollama), not the raw `transformers` path above. `llm.build_llm` returns a `ChatOpenAI` pointed at
that endpoint; `agent.build_math_agent` wraps it in `create_react_agent` with an `InMemorySaver`
checkpointer by default (reuse the same `{"configurable": {"thread_id": ...}}` config across turns of one
conversation to keep context).

`memory.strip_thinking_tags` is the string-based counterpart to `inference.solver.solve`'s token-id
splitting, for use when only decoded text is available (e.g. a `ChatOpenAI` response's `.content`). **It
is not wired into `build_math_agent` automatically** — call it yourself before writing an `AIMessage` back
into checkpointed history.

## `enrichment`

Adds a semantic layer on top of the Question/Answer/Tag graph from `data.build_graph_records`: reads each
question + its accepted answer with an LLM (structured JSON validated against the `GraphExtraction`
Pydantic schema in `schemas.py`) and attaches `Concept` nodes plus the solution's resolution steps. Never
creates Question/Answer/Tag nodes itself — it only adds to an existing `graph_data`.

**The graph has exactly three layers**, and the reason is worth stating because an earlier design got it
wrong twice (a `Domain` layer that duplicated `Tag`, then a `ResolutionStep` node layer):

| Layer | Node types | Role |
|---|---|---|
| Macro topic | `Tag` | Coarse categorization, community-canonicalized by StackExchange (`nt.number-theory`). Curated via `config.METADATA_TAGS`. |
| Substance | `Question`, `Answer` | The actual content. Resolution steps ride on the `Answer` as an ordered `resolution_steps` property. |
| Shared mid-layer | `Concept` | Fine-grained mathematical ideas that link problems across topics. |

A node type is only worth having if instances can be **shared** between questions. Resolution steps can't
— every solution's steps are private to it, and no traversal ever crosses from one question's chain to
another's — so they are node *properties*, not nodes. As node chains they were 39% of the graph while
connecting nothing.

Two interchangeable backends, both producing the same `GraphExtraction` shape:

| Backend | Client | `extract_fn` | `resolve_fn` | Requires |
|---|---|---|---|---|
| Gemini (default) | `build_gemini_client` | `extract_graph_entities` | `resolve_concepts_gemini` | `GEMINI_API_KEY` |
| Groq (`gpt-oss`) | `build_groq_client` | `extract_graph_entities_groq` | `resolve_concepts_groq` | `GROQ_API_KEY` |

`graph_enrichment.enrich_graph_records(graph_data, client=None, extract_fn=extract_graph_entities,
sleep_seconds=0, checkpoint_path=None, checkpoint_every=10)` is the batch entry point:

- `graph_data` can be an in-memory dict **or a path** to a JSON file saved by `save_graph_json` — a string
  is loaded automatically via `load_graph_json`.
- Questions that already have resolution steps attached are skipped, so calling this again on the same
  graph (e.g. re-running against a checkpoint file after a crash or a quota error) resumes rather than
  duplicating nodes.
- Swap providers via `extract_fn=extract_graph_entities_groq` + `client=build_groq_client()`;
  `sleep_seconds` paces requests between questions (e.g. `15` for Groq's free-tier 8K TPM limit — its own
  `extract_graph_entities_groq` also retries on `RateLimitError` with backoff, so pacing and retry work
  together).
- **Crash safety:** if `extract_fn` raises (e.g. a rate-limit retry finally exhausted),
  `graph_data` is saved to `checkpoint_path` *before* re-raising, and also every `checkpoint_every`
  successfully-enriched questions along the way — so a fatal error can't strand progress in a dead stack
  frame with no way to get it back. When `graph_data` is passed as a path, `checkpoint_path` defaults to
  that same path automatically; when `graph_data` is an in-memory dict and no `checkpoint_path` is given,
  nothing is saved to disk and a `UserWarning` is emitted (so a long run can't silently save nothing) —
  pass `checkpoint_path=` to enable iterative saving, or ignore the warning if you'll save the returned
  value yourself.
- **Controlled vocabulary:** the concept names already in the graph are fed to `extract_fn` as
  `known_concepts` each iteration, so the LLM reuses an existing name when one fits. A custom `extract_fn`
  must accept a `known_concepts` keyword (both built-ins do). This only helps at the margin (measured: it
  moved cross-question concept sharing from 4% to 10%) — **`concept_resolution` below is the pass that
  actually consolidates the layer.**

⚠️ When `graph_data` is a path, `checkpoint_path` defaults to that same path, so the input file is
overwritten in place. Pass an explicit `checkpoint_path` to keep a pristine raw graph to re-run from.

```python
from math_assistant_agent.enrichment import build_groq_client, extract_graph_entities_groq, enrich_graph_records

client = build_groq_client()
enrich_graph_records(
    "data/graph_math_raw.json",                  # raw input, left untouched
    checkpoint_path="data/graph_math_kb.json",   # enriched output
    client=client,
    extract_fn=extract_graph_entities_groq,
    sleep_seconds=15,
)
```

### `concept_resolution` — consolidating the Concept layer

Extraction runs one question at a time, so the same idea comes back phrased differently each time
("Kunneth Formula" / "Künneth theorem" / "Künneth's short exact sequence"). Left alone this is severe:
in a 100-question graph, **368 of 384 concepts attached to exactly one question**, so the layer meant to
link problems linked essentially nothing.

String normalization does not fix it — measured, lowercasing plus article/possessive stripping merged
**0 of 384**, because the duplication is semantic rather than orthographic. Prompting the extractor to
reuse names (above) helps only marginally. What works is a **separate global pass over every concept at
once**, using each concept's one-sentence `description` as disambiguation context — which is why
`schemas.Concept` carries a description at all.

```python
from math_assistant_agent.enrichment import resolve_concepts, apply_concept_resolution

alias_map = resolve_concepts(graph_data)          # LLM calls; returns {alias: canonical}
apply_concept_resolution(graph_data, alias_map)   # pure, no LLM — rewrites the graph
```

The two steps are separate on purpose: the map can be inspected (and hand-edited) before anything changes,
and re-applied without spending tokens again.

- `resolve_concepts(graph_data, client=None, resolve_fn=resolve_concepts_gemini, block_size=80)` resolves
  in blocks rather than one giant prompt, then runs a **second pass over the resulting canonicals** so
  duplicates that landed in different blocks still merge. Concepts with no `description` (e.g. extracted
  before the field existed) fall back to the titles of the questions they apply to.
- `build_alias_map(clusters, all_names)` maps any name the model omitted from every cluster to itself, so
  a concept can never silently vanish because the resolver forgot to mention it.
- `apply_concept_resolution` merges nodes (keeping the longest description, recording absorbed surface
  forms in an `aliases` property), repoints edges, and **deduplicates** them — two concepts merging while
  both pointed at the same question would otherwise leave a parallel edge behind.

## `visualization`

`pyvis_renderer.render_graph(graph_data, output_path="knowledge_graph.html")` — renders `graph_data` as an
interactive PyVis HTML page (nodes colored by label, edges labeled by relationship type). Open
`output_path` directly in a browser, no server needed.
