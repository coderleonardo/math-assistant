# `math_assistant_agent.data`

Data acquisition, HTML/LaTeX cleaning, and dataset formatting. `fetch_math_dataset` and
`clean_html_for_math` are the Phase 1 pipeline from `notebooks/modeling/math-agent-finetuning.ipynb`;
everything downstream of the raw data branches into two output formats depending on what you're
building:

- **SFT training data**: `fetch_math_dataset` → `prepare_dataset` (using `clean_html_for_math`
  internally) → `save_dataset_jsonl` — produces the ShareGPT JSONL consumed by
  [`training.md`](training.md).
- **Graph knowledge base**: `fetch_math_dataset` → `build_graph_records` (also using
  `clean_html_for_math`) → `save_graph_json` — produces node/edge records for the Phase 5 Graph RAG work
  (Neo4j-backed, not yet wired up — see below).

All public functions are re-exported from the package root, so `from math_assistant_agent.data import *`
(as used in `notebooks/data/ingestion.ipynb`) or explicit imports both work:

```python
from math_assistant_agent.data import fetch_math_dataset, prepare_dataset, build_graph_records
```

## `data.stackexchange.fetch_math_dataset`

```python
def fetch_math_dataset(api_key=None, num_questions=5)
```

Queries the StackExchange `search/advanced` endpoint on `mathoverflow.net`, filtered to **accepted**
answers sorted by votes, and returns a list of raw prompt/completion pairs.

- `api_key` — optional StackExchange API key. Anonymous requests are heavily rate-limited; pass a key for
  anything beyond a handful of test calls.
- `num_questions` — page size, i.e. how many questions to fetch.

Returns a list of dicts:

```python
{
    "question_id": 182412,
    "answer_id": 182433,
    "title": "Why do roots of polynomials tend to have absolute value close to 1?",
    "prompt": "<p>...raw HTML question body...</p>",
    "completion": "<p>...raw HTML accepted answer body...</p>",
    "tags": ["pr.probability", "polynomials", "cv.complex-variables"],
    "question_score": 446,
    "view_count": 69357,
    "link": "https://mathoverflow.net/questions/182412/...",
    "answer_score": 214,
}
```

`prompt`/`completion` are the only fields `prepare_dataset` uses. The rest —
`answer_id`, `tags`, `question_score`, `view_count`, `link`, `answer_score` — are consumed by
`build_graph_records` (below) to give the Answer node a stable, traceable id and to enrich the
Question/Answer/Tag nodes. All of these come from the current filter's response for free (`!6WPIomnMNcVD9`
already includes `tags`, `score`, `view_count`, and `link` on the question object, and `score` on each
answer) — no filter change was needed to expose them.

Prints progress/errors to stdout and returns `[]` on a non-200 response, matching the notebook's original
behavior — it does not raise.

```python
from math_assistant_agent.data.stackexchange import fetch_math_dataset

raw_items = fetch_math_dataset(num_questions=20)
# raw_items = fetch_math_dataset(api_key="YOUR_KEY", num_questions=100)  # for larger pulls
```

## `data.cleaning.clean_html_for_math`

```python
def clean_html_for_math(html_text)
```

Strips StackExchange's HTML down to plain text with BeautifulSoup, while **preserving MathJax/LaTeX**
(`$...$`) content — this is the one non-obvious invariant in the whole pipeline. Also:

- Inserts blank lines after block elements (`p`, `br`, `h1`–`h3`, `pre`, `div`) so paragraphs don't run
  together.
- Converts `<li>` items into `- ` bullet lines.
- Collapses 3+ consecutive newlines down to 2.

```python
from math_assistant_agent.data.cleaning import clean_html_for_math

html = "<p>Solve $x^2 + 1 = 0$ for <i>x</i>.</p>"
clean_html_for_math(html)
# "Solve $x^2 + 1 = 0$ for x."
```

`clean_html_for_math("")` and `clean_html_for_math(None)` both return `""` — no exception on empty input.

## `data.formatting.prepare_dataset`

```python
def prepare_dataset(dados, system_prompt=SYSTEM_PROMPT)
```

Converts the raw `{title, prompt, completion}` records from `fetch_math_dataset` into the ShareGPT-style
`{"messages": [...]}` format the Qwen3 chat template and TRL's `SFTTrainer` expect. Internally cleans
`prompt`/`completion` with `clean_html_for_math` and prefixes the user turn with the question title.

- `dados` — list of dicts as returned by `fetch_math_dataset` (must have `title`, `prompt`, `completion`
  keys).
- `system_prompt` — defaults to `config.SYSTEM_PROMPT`; override if you want a different persona baked
  into the training data.

```python
from math_assistant_agent.data.formatting import prepare_dataset

formatted = prepare_dataset(raw_items)
formatted[0]
# {
#     "messages": [
#         {"role": "system", "content": "Você é um assistente de matemática avançado..."},
#         {"role": "user", "content": "Título: ...\n\n<cleaned question text>"},
#         {"role": "assistant", "content": "<cleaned accepted-answer text>"},
#     ]
# }
```

## `data.formatting.save_dataset_jsonl`

```python
def save_dataset_jsonl(dataset_formatado, path="dataset_math_qlora.jsonl")
```

Writes the list of message dicts from `prepare_dataset` to a JSONL file (one JSON object per line,
`ensure_ascii=False` so Portuguese/LaTeX characters are written as-is). Returns the path it wrote to.

```python
from math_assistant_agent.data.formatting import save_dataset_jsonl

save_dataset_jsonl(formatted, path="dataset_math_qlora.jsonl")
# ✅ Dataset estruturado e salvo em: dataset_math_qlora.jsonl
```

## `data.graph.build_graph_records`

```python
def build_graph_records(dados)
```

Converts the raw records from `fetch_math_dataset` into a backend-agnostic node/edge representation for a
graph knowledge base: one `Question` node and one `Answer` node per item (linked by
`HAS_ACCEPTED_ANSWER`), plus one `Tag` node per unique tag string (linked to each tagged `Question` by
`TAGGED_WITH`). Text fields are cleaned with `clean_html_for_math`, same as `prepare_dataset`. Tag nodes
are deduplicated across the whole `dados` list — a tag shared by 50 questions still produces one `Tag`
node with 50 `TAGGED_WITH` edges pointing at it.

- `dados` — list of dicts as returned by `fetch_math_dataset` (needs `question_id`, `title`, `prompt`,
  `completion`; `answer_id`, `tags`, `question_score`, `view_count`, `link`, and `answer_score` are all
  optional and simply omitted/`None` from the resulting properties if absent).

Returns a single dict shaped `{"nodes": [...], "edges": [...]}`:

```python
{
    "nodes": [
        {
            "id": "question_182412",
            "label": "Question",
            "properties": {
                "question_id": 182412,
                "title": "Why do roots of polynomials tend to have absolute value close to 1?",
                "text": "<cleaned question text>",
                "score": 446,
                "view_count": 69357,
                "link": "https://mathoverflow.net/questions/182412/...",
            },
        },
        {
            "id": "answer_182433",
            "label": "Answer",
            "properties": {
                "answer_id": 182433,
                "question_id": 182412,
                "text": "<cleaned accepted-answer text>",
                "score": 214,
            },
        },
        {
            "id": "tag_pr.probability",
            "label": "Tag",
            "properties": {"name": "pr.probability"},
        },
    ],
    "edges": [
        {
            "source": "question_182412",
            "target": "answer_182433",
            "type": "HAS_ACCEPTED_ANSWER",
            "properties": {},
        },
        {
            "source": "question_182412",
            "target": "tag_pr.probability",
            "type": "TAGGED_WITH",
            "properties": {},
        },
    ],
}
```

`score`/`view_count`/`link` were chosen because they come back from the current filter (`!6WPIomnMNcVD9`)
at zero extra cost — no other endpoint fields (e.g. `owner`) are captured or modeled yet.

This shape doesn't commit to any specific graph database — `label`/`type` read naturally as Neo4j node
labels/relationship types if you go that route later (per the roadmap's planned Neo4j Graph RAG), but
nothing here depends on the `neo4j` driver, and the same records could just as easily be loaded into
`networkx` or any other graph library.

```python
from math_assistant_agent.data import fetch_math_dataset, build_graph_records

raw_items = fetch_math_dataset(api_key="YOUR_KEY", num_questions=200)
graph_data = build_graph_records(raw_items)
len(graph_data["nodes"]), len(graph_data["edges"])
# 200 Question + 200 Answer + N unique Tag nodes (N depends on tag reuse across questions);
# 200 HAS_ACCEPTED_ANSWER edges + one TAGGED_WITH edge per (question, tag) pair
```

## `data.graph.save_graph_json`

```python
def save_graph_json(graph_data, path="graph_math_kb.json")
```

Writes the `{"nodes": [...], "edges": [...]}` dict from `build_graph_records` to a single JSON file
(`ensure_ascii=False`, pretty-printed). Returns the path it wrote to.

```python
from math_assistant_agent.data import save_graph_json

save_graph_json(graph_data, path="graph_math_kb.json")
# ✅ Grafo de conhecimento salvo em: graph_math_kb.json
```

## `data.graph` query helpers

Three small read-only helpers for querying a `graph_data` dict (the same `{"nodes": [...], "edges":
[...]}` shape everything else in this module produces/consumes). These power the
[`enrichment`](enrichment.md) module and are handy for ad-hoc exploration:

```python
def get_node_by_id(graph_data, node_id)
def get_accepted_answer(graph_data, question_id)
def get_questions_by_min_score(graph_data, min_score=100)
```

- `get_node_by_id(graph_data, node_id)` — returns the full node dict for a given id, or `None`.
- `get_accepted_answer(graph_data, question_id)` — follows the `HAS_ACCEPTED_ANSWER` edge from a Question
  node id and returns the linked Answer node, or `None`.
- `get_questions_by_min_score(graph_data, min_score=100)` — returns all Question nodes with
  `properties["score"] > min_score`.

```python
from math_assistant_agent.data import get_node_by_id, get_accepted_answer, get_questions_by_min_score

question = get_node_by_id(graph_data, "question_182412")
answer = get_accepted_answer(graph_data, "question_182412")
popular = get_questions_by_min_score(graph_data, min_score=100)
```

## Full pipeline examples

**SFT training data:**

```python
from math_assistant_agent.data import fetch_math_dataset, prepare_dataset, save_dataset_jsonl

raw_items = fetch_math_dataset(api_key="YOUR_KEY", num_questions=200)
formatted = prepare_dataset(raw_items)
save_dataset_jsonl(formatted, path="dataset_math_qlora.jsonl")
```

The resulting `dataset_math_qlora.jsonl` is what `training.trainer.load_training_dataset` reads — see
[`training.md`](training.md).

**Graph knowledge base:**

```python
from math_assistant_agent.data import fetch_math_dataset, build_graph_records, save_graph_json

raw_items = fetch_math_dataset(api_key="YOUR_KEY", num_questions=200)
graph_data = build_graph_records(raw_items)
save_graph_json(graph_data, path="graph_math_kb.json")
```

Both branches can reuse the same `raw_items` — `fetch_math_dataset` only needs to be called once.

The graph knowledge base branch continues in [`enrichment.md`](enrichment.md) (adding Domain/Concept/
ResolutionStep nodes via Gemini) and [`visualization.md`](visualization.md) (rendering `graph_data` to
interactive HTML).
