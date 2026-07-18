from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_assistant_agent.config import MODEL_NAME
from math_assistant_agent.training.quantization import build_bnb_config


def load_finetuned_model(base_model_name=MODEL_NAME, adapter_path="./qwen-math-agent-adapter"):
    """Load the 4-bit base model, attach the LoRA adapter, and return (model, tokenizer).

    Uses device_map="auto" (unlike training, which forces device_map={"": 0} to avoid
    DataParallel conflicts with SFTTrainer). Sets the model to eval mode.

    Example:
        model, tokenizer = load_finetuned_model(adapter_path="./qwen-math-agent-adapter")
    """
    print("1. Configurando ambiente e carregando Tokenizer...")
    bnb_config = build_bnb_config()

    # Carregamos o tokenizer que foi salvo junto com o adaptador
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)

    print("2. Carregando Modelo Base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Agora podemos voltar para 'auto' para inferência!
        torch_dtype="auto",
    )

    print("3. Acoplando o seu Agente (Adaptador LoRA)...")
    # Une o modelo base com o que ele aprendeu no StackExchange
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()  # Coloca o modelo explicitamente em modo de avaliação/inferência
    print("✅ Modelo pronto para uso!")

    return model, tokenizer
