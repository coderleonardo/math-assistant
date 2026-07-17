import re


def strip_thinking_tags(texto):
    """Remove tudo que estiver entre as tags <think> e </think>."""
    # O regex remove a tag e seu conteúdo interno
    texto_limpo = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
    return texto_limpo.strip()
