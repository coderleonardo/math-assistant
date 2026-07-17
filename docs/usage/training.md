# `math_assistant_agent.training`

QLoRA fine-tuning setup — quantization config, LoRA config, model/tokenizer loading, and the
`SFTTrainer` wiring from Phase 2 of `notebooks/modeling/math-agent-finetuning.ipynb`.

Requires a JSONL dataset already produced by [`data.formatting`](data.md), e.g.
`dataset_math_qlora.jsonl`.

## `training.quantization.build_bnb_config`

```python
def build_bnb_config()
```

Returns a `transformers.BitsAndBytesConfig` for 4-bit NF4 quantization with double quantization and
`bfloat16` compute dtype — the "Q" in QLoRA. Takes no arguments; this is the exact config the notebook
uses both for training and for reloading the base model at inference time (see
[`inference.model_loading`](inference.md)).

```python
from math_assistant_agent.training.quantization import build_bnb_config

bnb_config = build_bnb_config()
```

## `training.lora.build_lora_config`

```python
def build_lora_config()
```

Returns a `peft.LoraConfig` targeting all attention and MLP projections
(`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), `r=16`, `lora_alpha=32`,
`lora_dropout=0.05`, `task_type="CAUSAL_LM"`. Takes no arguments.

```python
from math_assistant_agent.training.lora import build_lora_config

lora_config = build_lora_config()
```

## `training.model_loading`

```python
def load_tokenizer(model_name=MODEL_NAME)
def load_base_model_for_training(model_name=MODEL_NAME, bnb_config=None)
```

- `load_tokenizer` loads the tokenizer for `model_name` (defaults to `config.MODEL_NAME`) and sets
  `pad_token = eos_token` if the tokenizer has no pad token — required before training.
- `load_base_model_for_training` loads the base causal LM 4-bit quantized, **hardcoded to
  `device_map={"": 0}`**. This is a deliberate workaround: `SFTTrainer` conflicts with `DataParallel` on
  multi-GPU machines, so training must be pinned to a single device. If `bnb_config` isn't supplied it
  calls `build_bnb_config()` for you. (Inference uses `device_map="auto"` instead — see
  [`inference.model_loading`](inference.md).)

```python
from math_assistant_agent.training.model_loading import load_tokenizer, load_base_model_for_training

tokenizer = load_tokenizer()
model = load_base_model_for_training()
```

## `training.trainer`

```python
def load_training_dataset(jsonl_path)
def build_training_args(output_dir="./qwen-math-agent", **overrides)
def build_trainer(model, training_args, dataset, tokenizer, peft_config)
def train_and_save_adapter(trainer, tokenizer, adapter_dir="./qwen-math-agent-adapter")
```

- `load_training_dataset(jsonl_path)` — thin wrapper over `datasets.load_dataset("json", ...,
  split="train")`.
- `build_training_args(output_dir=..., **overrides)` — returns a `trl.SFTConfig` with the notebook's
  defaults (`per_device_train_batch_size=1`, `gradient_accumulation_steps=4`, `learning_rate=2e-4`,
  `max_steps=1`, `optim="paged_adamw_8bit"`, `dataset_text_field="messages"`, `max_length=2048`,
  `packing=False`, `fp16`/`bf16` auto-selected via `torch.cuda.is_bf16_supported()`). Pass any `SFTConfig`
  keyword as an override, e.g. `num_train_epochs=3` to replace the notebook's smoke-test `max_steps=1`.
- `build_trainer(...)` — thin wrapper over `trl.SFTTrainer`, wiring `model`, `training_args`, `dataset`,
  `tokenizer` (as `processing_class`), and `peft_config` together. TRL wraps `model` with the LoRA config
  internally.
- `train_and_save_adapter(trainer, tokenizer, adapter_dir=...)` — calls `trainer.train()`, then saves
  **only the LoRA adapter** (a few MB) plus the tokenizer to `adapter_dir`. Not the full base model.

## Full pipeline example

```python
from math_assistant_agent.training.quantization import build_bnb_config
from math_assistant_agent.training.lora import build_lora_config
from math_assistant_agent.training.model_loading import load_tokenizer, load_base_model_for_training
from math_assistant_agent.training.trainer import (
    load_training_dataset,
    build_training_args,
    build_trainer,
    train_and_save_adapter,
)

tokenizer = load_tokenizer()
model = load_base_model_for_training(bnb_config=build_bnb_config())
lora_config = build_lora_config()

dataset = load_training_dataset("dataset_math_qlora.jsonl")
training_args = build_training_args(num_train_epochs=3)  # override the notebook's max_steps=1 smoke test

trainer = build_trainer(model, training_args, dataset, tokenizer, lora_config)
adapter_dir = train_and_save_adapter(trainer, tokenizer)
# adapter_dir == "./qwen-math-agent-adapter"
```

The saved adapter directory is what [`inference.model_loading.load_finetuned_model`](inference.md) loads
back for generation.
