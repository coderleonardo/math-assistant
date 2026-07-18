# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

This is an early-stage ML research project, not a packaged application. There is no `src/` layout,
no `requirements.txt`/`pyproject.toml`, no test suite, and no lint/CI config. All working code currently
lives inside a single Jupyter notebook plus one standalone example script:

- `notebooks/modeling/math-agent-finetuning.ipynb` — the actual, runnable pipeline (data acquisition →
  cleaning → QLoRA fine-tuning → merge → inference smoke test).
- `notebooks/modeling/conversational_agent_example.py` — a standalone design sketch for the LangGraph
  agent-serving layer. It targets a locally-hosted OpenAI-compatible endpoint (vLLM/Ollama) and is **not**
  wired into the notebook; treat it as a reference for Phase 4 work, not working production code.
- `docs/_01_phases.md` — Portuguese-language roadmap, mirrors the "Roadmap & Implementation Phases"
  section of `README.md`. Check both if phase status looks stale — they are maintained by hand and can drift.

Because there's no package structure yet, don't invent one (no `src/`, `pyproject.toml`, test scaffolding,
etc.) unless the user asks for it — confirm the intended layout first since this is a green-field decision.

## Working with the notebook

`math-agent-finetuning.ipynb` cells must run in order — later cells depend on variables from earlier ones
(e.g. `model`, `tokenizer`, `lora_config`, `bnb_config` from the setup cells; `meu_dataset` /
`meu_dataset_pronto` from the extraction/cleaning cells). There's no `requirements.txt`; dependencies are
installed inline via `!pip install` cells at the top of the notebook:
```
transformers peft bitsandbytes accelerate beautifulsoup4 trl datasets
```
Editing/inspecting it programmatically: it's stored as a single-line JSON file (no line terminators), so
prefer `NotebookEdit`/`nbformat`-aware tools or `python3 -c "import json; ..."` over line-based tools like
`sed`/`grep -n`.

## Architecture — the pipeline, end to end

1. **Data acquisition** — StackExchange `search/advanced` API (Math StackExchange / MathOverflow), filtered
   to accepted, highly-voted answers. See `fetch_math_dataset()` in the notebook.
2. **Cleaning** — `clean_html_for_math()` strips HTML with BeautifulSoup while deliberately preserving
   MathJax/LaTeX (`$...$`) content — this is the one non-obvious invariant in the cleaning step; don't let a
   generic HTML-stripping change eat LaTeX delimiters.
3. **Dataset formatting** — `prepare_dataset()` converts cleaned Q/A pairs into ShareGPT-style
   `{"messages": [{"role": "system"|"user"|"assistant", ...}]}` JSONL, written to `dataset_math_qlora.jsonl`.
   This is the format the Qwen3 chat template and TRL's `SFTTrainer` expect.
4. **QLoRA fine-tuning** — base model `Qwen/Qwen3-4B-Thinking-2507` loaded 4-bit (`BitsAndBytesConfig`,
   NF4, double-quant, bfloat16 compute), LoRA applied to all attention/MLP projections
   (`q/k/v/o_proj`, `gate/up/down_proj`), trained with TRL's `SFTTrainer`. Only the adapter is saved
   (`./qwen-math-agent-adapter`), not the full model.
   - **Multi-GPU gotcha**: training must force `device_map={"": 0}` to avoid `DataParallel` conflicts with
     `SFTTrainer`. Inference/merge code can use `device_map="auto"` instead.
5. **Merge & deploy** — base model reloaded in `bfloat16`, merged with the LoRA adapter
   (`merge_and_unload`) to produce a standalone model, then served via vLLM or Ollama behind an
   OpenAI-compatible API for the agent layer to call.
6. **Agent orchestration (Phase 4, in progress)** — `conversational_agent_example.py` shows the intended
   shape: `langchain_openai.ChatOpenAI` pointed at the local inference server, wrapped by LangGraph's
   `create_react_agent`, with `InMemorySaver` as the short-term memory checkpointer.
7. **Graph knowledge base (in progress)**: `src/math_assistant_agent/data/graph.py` structures raw
   StackExchange data into a Question/Answer/Tag graph (`build_graph_records`); `enrichment/` calls an
   LLM with structured JSON output to extract a `Concept -> ResolutionStep` "Graph Chain of
   Thought" on top of it (`enrich_graph_records`); `visualization/` renders the result with PyVis
   (`render_graph`). The extraction backend is pluggable via `enrich_graph_records`'s `extract_fn` param
   — `extract_graph_entities` (Gemini, `response_schema`, the default) or `extract_graph_entities_groq`
   (Groq `gpt-oss` models, JSON mode + `tenacity` retry on rate limits); both produce the same
   `GraphExtraction` shape, so `enrich_graph_with_entities`'s node/edge building never varies by
   provider. See `docs/usage.md`. The Neo4j database itself is
   not wired up yet — everything above operates on a backend-agnostic `{"nodes": [...], "edges": [...]}`
   dict, persisted as JSON.
8. **Planned (not yet implemented)**: Neo4j-backed persistence/retrieval for the graph above, Python
   sandbox / Wolfram Alpha tool calling, Redis semantic caching, and teacher→student distillation to a
   smaller (0.5B–1.5B) model.

## Model-specific rules (Qwen3-4B-Thinking-2507)

These are load-bearing for correctness, not stylistic preferences:

- **System prompt** must include: *"Please reason step by step, and put your final answer within
  \boxed{}."* — required for the model's expected output format.
- **Recommended generation params**: `temperature=0.6`, `top_p=0.95`, `top_k=20`,
  `max_new_tokens=32768` (up to `81920` for very hard problems).
- **Thinking-tag stripping is mandatory for multi-turn history.** The model emits reasoning inside
  `<think>...</think>` before the final answer. When persisting conversation history (e.g. via the
  LangGraph checkpointer), only the final-answer portion may be fed back into subsequent turns —
  including raw `<think>` content in history degrades reasoning quality. The notebook's inference code
  splits thinking from the final answer by locating token id `151668` (the `</think>` close tag) in the
  generated ids, not by string matching — reuse that approach rather than a naive `<think>` regex on
  decoded text when working with raw generation output; a regex-based fallback is acceptable when only
  decoded text is available (see the commented `limpar_raciocinio()` helper at the bottom of
  `conversational_agent_example.py`).
