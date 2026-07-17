# `math_assistant_agent.agent`

LangGraph agent-serving layer, extracted from
`notebooks/modeling/conversational_agent_example.py`. This is Phase 4 (in progress): it targets a
**locally-hosted OpenAI-compatible endpoint** (vLLM or Ollama serving the fine-tuned model), not the raw
`transformers` generation path used by [`inference.md`](inference.md). You need an inference server
running at `api_base` before any of this will actually respond.

## `agent.llm.build_llm`

```python
def build_llm(
    model="meu-agente-matematico-final",
    api_base="http://localhost:8000/v1",
    api_key="sk-local",
    temperature=0.6,
    top_p=0.95,
)
```

Returns a `langchain_openai.ChatOpenAI` pointed at your local inference server.

- `model` — the model name you deployed under (vLLM/Ollama model tag), not a HuggingFace repo id.
- `api_base` — your OpenAI-compatible endpoint, e.g. vLLM's `http://localhost:8000/v1`.
- `api_key` — a placeholder string; local servers typically don't check it.
- `temperature` / `top_p` — Qwen3's recommended sampling params (`top_p` is passed through
  `model_kwargs` since `ChatOpenAI` doesn't expose it directly).

```python
from math_assistant_agent.agent.llm import build_llm

llm = build_llm()
# llm = build_llm(model="qwen-math-agent", api_base="http://localhost:11434/v1")  # e.g. Ollama
```

## `agent.agent.build_math_agent`

```python
def build_math_agent(llm, tools=None, system_prompt=AGENT_SYSTEM_PROMPT, checkpointer=None)
```

Wraps `langgraph.prebuilt.create_react_agent`. Defaults `tools` to `[]` (no tools wired up yet — Phase 5
is still planned) and `checkpointer` to a fresh `InMemorySaver()` if not supplied. `system_prompt` defaults
to `config.AGENT_SYSTEM_PROMPT`, which includes the `\boxed{}` instruction Qwen3 requires.

```python
from math_assistant_agent.agent.llm import build_llm
from math_assistant_agent.agent.agent import build_math_agent

llm = build_llm()
math_agent = build_math_agent(llm)

config = {"configurable": {"thread_id": "math-session-001"}}

from langchain_core.messages import HumanMessage

response = math_agent.invoke(
    {"messages": [HumanMessage(content="Qual é a regra da cadeia em cálculo?")]},
    config,
)
print(response["messages"][-1].content)
```

`InMemorySaver` keys history off `thread_id` — reuse the same `config` dict across turns of one
conversation to keep context, and use a different `thread_id` to start a fresh one.

## `agent.memory.strip_thinking_tags`

```python
def strip_thinking_tags(texto)
```

Regex-based removal of `<think>...</think>` content from a decoded string:
`re.sub(r'<think>.*?</think>', '', texto, flags=re.DOTALL)`. This is the string-based counterpart to the
token-id approach in `inference.solver.solve` — use this one when you only have decoded text to work with
(e.g. a `ChatOpenAI` response's `.content`), since the agent-serving path here doesn't have access to raw
generated token ids the way the direct `transformers` inference path does.

**Multi-turn history is not safe without this.** The model emits reasoning inside `<think>...</think>`
before its final answer; feeding raw `<think>` content back into subsequent turns (e.g. via the
checkpointer) degrades reasoning quality. Before saving an `AIMessage` into history, strip it:

```python
from langchain_core.messages import AIMessage
from math_assistant_agent.agent.memory import strip_thinking_tags

raw_reply = response["messages"][-1]
clean_reply = AIMessage(content=strip_thinking_tags(raw_reply.content))
```

This function isn't wired into `build_math_agent` automatically — the original script only sketches this
as manual state-update logic (no custom LangGraph node calls it yet). Apply it yourself wherever you
persist agent output into history.
