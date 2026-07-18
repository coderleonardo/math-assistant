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
from math_assistant_agent.enrichment import enrich_graph_records
from math_assistant_agent.visualization import render_graph

raw_items = fetch_math_dataset(num_questions=50)

# Fine-tuning branch
save_dataset_jsonl(prepare_dataset(raw_items), path="dataset_math_qlora.jsonl")

# Knowledge-graph branch
graph_data = build_graph_records(raw_items)
save_graph_json(graph_data, path="data/graph_math_kb.json")

enrich_graph_records(graph_data)  # Gemini by default; see enrichment.md below
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
Pydantic schema in `schemas.py`) and extracts `Concept -> Question -> ResolutionStep`, a "Graph Chain of
Thought." The StackExchange `Tag` layer already handles coarse categorization (community-canonicalized,
e.g. `nt.number-theory`), so there is deliberately no separate domain layer. Never creates
Question/Answer/Tag nodes itself — it only adds to an existing `graph_data`.

Two interchangeable backends, both producing the same `GraphExtraction` shape:

| Backend | Client | `extract_fn` | Requires |
|---|---|---|---|
| Gemini (default) | `build_gemini_client` | `extract_graph_entities` | `GEMINI_API_KEY` |
| Groq (`gpt-oss`) | `build_groq_client` | `extract_graph_entities_groq` | `GROQ_API_KEY` |

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
  `known_concepts` each iteration, so the LLM reuses an existing name when one fits instead of inventing a
  new phrasing for the same idea — this is what keeps the `Concept` layer connective rather than near-1:1
  with questions. A custom `extract_fn` must accept a `known_concepts` keyword (both built-ins do). String
  normalization is deliberately *not* used for this — concept duplication here is semantic, not spelling,
  so it merges nothing; embedding-based merging of an existing graph's concepts is the future lever for
  retro-consolidation.

```python
from math_assistant_agent.enrichment import build_groq_client, extract_graph_entities_groq, enrich_graph_records

client = build_groq_client()
enrich_graph_records(
    "data/graph_math_kb.json", client=client, extract_fn=extract_graph_entities_groq, sleep_seconds=15
)
```

## `visualization`

`pyvis_renderer.render_graph(graph_data, output_path="knowledge_graph.html")` — renders `graph_data` as an
interactive PyVis HTML page (nodes colored by label, edges labeled by relationship type). Open
`output_path` directly in a browser, no server needed.
