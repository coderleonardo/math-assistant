# `math_assistant_agent.config`

Shared constants used across the `data`, `training`, `inference`, and `agent` modules. This module has
no third-party dependencies — it's safe to import on its own.

## Constants

| Name | Value | Used by |
|---|---|---|
| `MODEL_NAME` | `"Qwen/Qwen3-4B-Thinking-2507"` | `training.model_loading`, `inference.model_loading` |
| `THINK_END_TOKEN_ID` | `151668` | `inference.solver` — token id of the closing `</think>` tag, used to split reasoning from the final answer |
| `SYSTEM_PROMPT` | Portuguese persona string used during fine-tuning | `data.formatting.prepare_dataset`, `inference.solver.solve` |
| `AGENT_SYSTEM_PROMPT` | `SYSTEM_PROMPT` + the `\boxed{}` instruction required by Qwen3 | `agent.agent.build_math_agent` |
| `GENERATION_DEFAULTS` | `{"max_new_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20}` | `inference.solver.solve` |

## Example

```python
from math_assistant_agent.config import MODEL_NAME, GENERATION_DEFAULTS

print(MODEL_NAME)
# "Qwen/Qwen3-4B-Thinking-2507"

print(GENERATION_DEFAULTS)
# {"max_new_tokens": 4096, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
```

`SYSTEM_PROMPT` and `AGENT_SYSTEM_PROMPT` are two different strings on purpose:

- `SYSTEM_PROMPT` — used at fine-tuning time and by the notebook-style `solve()` inference helper. It does
  **not** include the `\boxed{}` instruction, matching what the model was actually trained on.
- `AGENT_SYSTEM_PROMPT` — used by the LangGraph agent (`agent.agent.build_math_agent`), which talks to the
  model over an OpenAI-compatible API rather than raw `generate()`. It adds Qwen3's required
  *"Please reason step by step, and put your final answer within \boxed{}."* instruction.

If you fine-tune with a different persona, update `SYSTEM_PROMPT` here rather than editing the string in
`data.formatting` or `inference.solver` directly — both import it from this module.
