# 🧠 AI Math Assistant Agent

General-purpose LLMs often produce math answers that *look* right but skip steps, hallucinate
intermediate results, or can't justify how they got from problem to answer. This project builds an
assistant agent that **grounds** its reasoning instead of just asserting it — by structuring real,
community-vetted solved problems into a knowledge graph and retrieving relevant context before the
model answers.

> **Status:** early-stage research project. The data pipeline, knowledge-graph construction, and a
> working retrieval-augmented agent are implemented. Fine-tuning, persistence (Neo4j), tool calling,
> and distillation are on the roadmap below.

---

## 🚀 Overview

The agent is built from three layers:

1. **Real, vetted data.** Accepted, highly-voted answers are pulled from math Q&A communities and
   cleaned while preserving their LaTeX.
2. **A knowledge graph, not just a black box.** Each question/answer pair is structured into a
   backend-agnostic graph of `Question`, `Answer`, and `Tag` nodes, so solutions can be retrieved and
   reused rather than regenerated from scratch every time.
3. **Agentic orchestration.** A LangGraph state machine extracts keywords from the user's question,
   retrieves matching solved problems from the graph, and generates a final answer grounded **only** in
   that retrieved context.

---

## 🧩 Architecture

```
StackExchange API ──► clean HTML (preserve LaTeX) ──► build graph ──► JSON ──► PyVis visualization
                                                  (Question / Answer / Tag)
                                                          │
                                                          ▼
                                    LangGraph agent:  extractor ──► retriever ──► brain
                                    (keywords)        (graph lookup)   (grounded answer, Groq)
```

- **Data acquisition** — `fetch_math_dataset()` pulls accepted, highly-voted answers from the
  StackExchange `search/advanced` API (MathOverflow / Mathematics).
- **Cleaning** — `clean_html_for_math()` strips HTML with BeautifulSoup while deliberately preserving
  the MathJax/LaTeX (`$...$`) inside it.
- **Knowledge graph** — `build_graph_records()` produces a `{"nodes": [...], "edges": [...]}` dict
  (`Question`/`Answer`/`Tag` nodes; `HAS_ACCEPTED_ANSWER`/`TAGGED_WITH` edges), persisted as JSON and
  rendered with `render_graph()` (PyVis).
- **Agent** — `build_agent(client, graph_data)` compiles a LangGraph `StateGraph` with three nodes:
  keyword `extractor` → graph `retriever` → answer `generator` ("brain"), backed by Groq.

---

## 📂 Repository structure

```
src/math_assistant_agent/
├── config.py              # model names, prompts, generation defaults, tag curation
├── data/                  # StackExchange fetch, HTML cleaning, dataset formatting, graph build/IO
├── visualization/         # PyVis renderer
└── agent/                 # LangGraph agent
    ├── states.py          #   GraphAgentState
    ├── llm.py             #   Groq client + chat-completion helper
    ├── extractor.py       #   node 1: question -> keywords
    ├── retriever.py       #   node 2: keywords -> graph context
    ├── generator.py       #   node 3: context -> grounded answer
    └── flow.py            #   build_agent(): wires the nodes into a StateGraph

scripts/mvp.py             # fast CLI to run the agent on a single question
notebooks/                 # data ingestion, knowledge-graph, and modeling notebooks
docs/                      # usage notes and roadmap
```

---

## 🛠️ Tech stack

| Layer | Tools |
|---|---|
| Data sourcing | StackExchange API, `requests`, `beautifulsoup4` |
| Knowledge graph | Backend-agnostic node/edge dict (JSON), `pyvis` for visualization |
| Agent orchestration | `langgraph`, `langchain-core`, `pydantic` |
| LLM provider (agent) | Groq (`openai/gpt-oss-20b`) |

---

## ⚡ Getting started

**Prerequisites:** Python ≥ 3.11. [`uv`](https://docs.astral.sh/uv/) is recommended (a `uv.lock` is
committed); plain `pip` works too.

```bash
# 1. Install the package and its dependencies
uv sync                       # or:  pip install -e .

# 2. Configure API keys
cp .env.example .env          # then fill in the values you need
```

`.env` keys (see `.env.example`): `STACK_EXCHANGE_API_KEY` (data acquisition), `GROQ_API_KEY` (required
by the agent). The agent needs a knowledge-graph JSON to retrieve from; graph artifacts are generated
locally (they are gitignored, not shipped) — build one with the ingestion notebook under
`notebooks/data/` first.

### Run the agent

```bash
python scripts/mvp.py
python scripts/mvp.py "How do I integrate x^2?"
python scripts/mvp.py "How do I integrate x^2?" --graph data/graph_math_2026-07-17-16.json
```

`scripts/mvp.py` loads the graph, builds the agent, and prints the extracted keywords, the retrieval
result, and the final grounded answer.

### Build the knowledge graph

Use `notebooks/data/ingestion.ipynb` to fetch data and build the graph JSON the agent retrieves from.

---

## 🗺️ Roadmap

Planned, not yet implemented:

- Supervised fine-tuning (QLoRA) of a compact open-weight model on the collected data.
- Neo4j-backed persistence and retrieval for the knowledge graph (currently JSON on disk).
- Tool calling (Python sandbox / Wolfram Alpha) so the agent can verify computations.
- Semantic caching for repeated queries.
- Teacher→student distillation to a smaller (0.5B–1.5B) model.

---

## 🤝 Acknowledgments

* **[StackExchange API](https://api.stackexchange.com/)** for the mathematical data.
* **HuggingFace** for the `trl` and `peft` ecosystems.

---

## 📄 License

This project's code is licensed under the [MIT License](LICENSE).

Content sourced from Math StackExchange / MathOverflow (questions, answers, and anything derived from
them, such as the training dataset and knowledge graph) is licensed by StackExchange under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), per their
[Terms of Service](https://stackoverflow.com/legal/terms-of-service/public). That license requires
attribution and share-alike for redistributed data — it applies independently of this repo's MIT code
license.
