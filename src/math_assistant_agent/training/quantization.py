import torch
from transformers import BitsAndBytesConfig


def build_bnb_config():
    """Configuração do BitsAndBytes para quantização em 4-bits (o "Q" do QLoRA)."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use float16 se sua GPU não suportar bfloat16
    )
