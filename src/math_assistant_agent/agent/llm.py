"""Provider-pluggable LLM plumbing for the agent nodes.

`get_llm_client` builds the provider SDK client; `llm_response` issues a single
chat completion. Only Groq is implemented today; the `provider_name` branch keeps
the door open for other backends.
"""

from groq import Groq

from math_assistant_agent.config import GROQ_EXTRACTION_TEMPERATURE, GROQ_MODEL_NAME


def get_llm_client(provider_name: str = "groq", api_key: str = ""):
    if provider_name == "groq":
        return Groq(api_key=api_key)
    else:
        return "Other client not yet implemented"


def llm_response(
    client,
    provider_name: str = "",
    system_prompt: str = "",
    question: str = "",
    model: str = GROQ_MODEL_NAME,
    temperature: float = GROQ_EXTRACTION_TEMPERATURE,
    json_mode: bool = False,
):
    if provider_name == "groq":
        # Preparamos os argumentos base
        kwargs = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            "temperature": temperature,
            "model": model,
        }

        # Injetamos o JSON mode APENAS se for solicitado
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # Fazemos a chamada desembrulhando os kwargs
        chat_completion = client.chat.completions.create(**kwargs)

        content = chat_completion.choices[0].message.content
        return content
    else:
        print("Provided model not implemented yet")
        return ""
