import json

from math_assistant_agent.config import SYSTEM_PROMPT
from math_assistant_agent.data.cleaning import clean_html_for_math


def prepare_dataset(dados, system_prompt=SYSTEM_PROMPT):
    """Convert fetch_math_dataset items into ShareGPT-style {"messages": [...]} records.

    Each record has a system/user/assistant turn, matching what TRL's SFTTrainer and
    the Qwen3 chat template expect. HTML in prompt/completion is cleaned via
    clean_html_for_math first.

    Example:
        >>> formatted = prepare_dataset(raw_items)
        >>> formatted[0]["messages"][0]["role"]
        'system'
    """
    dataset_formatado = []

    for item in dados:
        # 1. Limpar os textos
        pergunta_limpa = clean_html_for_math(item["prompt"])
        resposta_limpa = clean_html_for_math(item["completion"])

        # 2. Juntar o título com o corpo da pergunta para dar mais contexto ao modelo
        conteudo_usuario = f"Título: {item['title']}\n\n{pergunta_limpa}"

        # 3. Montar a estrutura de mensagens
        conversacao = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conteudo_usuario},
                {"role": "assistant", "content": resposta_limpa},
            ]
        }

        dataset_formatado.append(conversacao)

    return dataset_formatado


def save_dataset_jsonl(dataset_formatado, path="dataset_math_qlora.jsonl"):
    """Write dataset_formatado (from prepare_dataset) to path as JSON Lines. Returns path."""
    with open(path, "w", encoding="utf-8") as f:
        for registro in dataset_formatado:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    print(f"✅ Dataset estruturado e salvo em: {path}")
    return path
