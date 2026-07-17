# `math_assistant_agent.inference`

Loading the fine-tuned model for local generation and running it through `model.generate()` directly
(no serving layer) — Phase 3's inference smoke test from
`notebooks/modeling/math-agent-finetuning.ipynb`.

This is the "raw `transformers` generate" path. If you're serving the merged model behind an
OpenAI-compatible API (vLLM/Ollama) instead, see [`agent.md`](agent.md) — that path doesn't use this
module.

## `inference.model_loading.load_finetuned_model`

```python
def load_finetuned_model(base_model_name=MODEL_NAME, adapter_path="./qwen-math-agent-adapter")
```

Loads the base model 4-bit quantized (via `training.quantization.build_bnb_config`) with
`device_map="auto"`, then attaches the trained LoRA adapter with `PeftModel.from_pretrained` and puts
the model in `.eval()` mode. The tokenizer is loaded from `adapter_path`, not `base_model_name` — the
adapter directory is where `training.trainer.train_and_save_adapter` saved it alongside the adapter
weights.

Note this attaches the adapter without merging it (`merge_and_unload` is not called) — matches what the
notebook actually does. Returns `(model, tokenizer)`.

```python
from math_assistant_agent.inference.model_loading import load_finetuned_model

model, tokenizer = load_finetuned_model()
# model, tokenizer = load_finetuned_model(adapter_path="./qwen-math-agent-adapter")
```

## `inference.solver.solve`

```python
def solve(pergunta, model, tokenizer, system_prompt=SYSTEM_PROMPT, **generation_overrides)
```

Runs one generation turn: applies the Qwen3 chat template, generates with
`config.GENERATION_DEFAULTS` (`max_new_tokens=4096, temperature=0.6, top_p=0.95, top_k=20` — override any
of these via keyword args), then splits the output into `(thinking_content, final_answer)` by locating
`config.THINK_END_TOKEN_ID` (`151668`, the `</think>` close tag) in the generated token ids — not by
string-matching the decoded text. If the tag isn't found, `thinking_content` falls back to a placeholder
string and the whole output is returned as `final_answer`.

```python
from math_assistant_agent.inference.model_loading import load_finetuned_model
from math_assistant_agent.inference.solver import solve

model, tokenizer = load_finetuned_model()

thinking, answer = solve("Como eu provo o teorema do sanduíche de análise?", model, tokenizer)
print(answer)

# Override generation params for a harder problem:
thinking, answer = solve(
    "Prove que a soma dos ângulos internos de um triângulo é 180°.",
    model,
    tokenizer,
    max_new_tokens=8192,
)
```

**Important:** `thinking` is the model's raw `<think>...</think>` reasoning. If you're persisting this
exchange into multi-turn conversation history, only `answer` should be fed back into subsequent turns —
see `agent.memory.strip_thinking_tags` in [`agent.md`](agent.md) for the string-based equivalent used on
the agent-serving side.
