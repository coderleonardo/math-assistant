from peft import LoraConfig


def build_lora_config():
    """Build the LoraConfig applied to all attention/MLP projections (r=16, alpha=32)."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
