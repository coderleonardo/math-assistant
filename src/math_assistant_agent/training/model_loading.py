from transformers import AutoModelForCausalLM, AutoTokenizer

from math_assistant_agent.config import MODEL_NAME
from math_assistant_agent.training.quantization import build_bnb_config


def load_tokenizer(model_name=MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # É importante definir o token de preenchimento (pad_token) para o treinamento
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model_for_training(model_name=MODEL_NAME, bnb_config=None):
    if bnb_config is None:
        bnb_config = build_bnb_config()

    # device_map={"": 0} força o carregamento em uma única GPU, contornando
    # conflitos de DataParallel do SFTTrainer com múltiplas GPUs.
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype="auto",
    )
