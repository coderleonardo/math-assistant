from langchain_openai import ChatOpenAI


def build_llm(
    model="meu-agente-matematico-final",  # O nome que você deu ao fazer o deploy local
    api_base="http://localhost:8000/v1",  # URL do seu motor de inferência (vLLM/Ollama)
    api_key="sk-local",  # Chave fictícia para uso local
    temperature=0.6,  # Parâmetro recomendado para o Qwen3
    top_p=0.95,  # Top-P recomendado
):
    """Build a ChatOpenAI client pointed at a local OpenAI-compatible inference server.

    Defaults match a merged, self-hosted model served via vLLM/Ollama, not a hosted API
    — api_key is a placeholder, not a real credential.

    Example:
        llm = build_llm(api_base="http://localhost:8000/v1")
    """
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=api_base,
        temperature=temperature,
        model_kwargs={"top_p": top_p},
    )
