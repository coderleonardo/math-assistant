import torch

from math_assistant_agent.config import GENERATION_DEFAULTS, SYSTEM_PROMPT, THINK_END_TOKEN_ID


def solve(pergunta, model, tokenizer, system_prompt=SYSTEM_PROMPT, **generation_overrides):
    # Usamos o mesmo system prompt do treinamento para ativar a "persona" correta
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": pergunta},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generation_kwargs = dict(GENERATION_DEFAULTS)
    generation_kwargs.update(generation_overrides)

    print("Pensando...\n")
    with torch.no_grad():  # Desativa cálculo de gradientes (economiza VRAM e CPU)
        generated_ids = model.generate(**model_inputs, **generation_kwargs)

    # Isolar apenas a resposta gerada (tirando o prompt de entrada)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # Fazendo o parse do conteúdo de "Thinking" (A tag de fechamento é o token THINK_END_TOKEN_ID)
    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        final_answer = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    except ValueError:
        thinking_content = "Não foi possível separar o raciocínio."
        final_answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    return thinking_content, final_answer
