import re


def strip_thinking_tags(texto):
    """Remove any <think>...</think> block from texto.

    Regex-based fallback for when only decoded text is available; prefer splitting on
    the </think> token id (151668) when raw generation output is at hand (see
    CLAUDE.md's "Thinking-tag stripping" note).

    Example:
        >>> strip_thinking_tags("<think>reasoning</think>final answer")
        'final answer'
    """
    # O regex remove a tag e seu conteúdo interno
    texto_limpo = re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL)
    return texto_limpo.strip()
