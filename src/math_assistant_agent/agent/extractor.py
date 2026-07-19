"""Node 1: read the user's question and extract mathematical keywords."""

import json

from math_assistant_agent.agent.llm import llm_response
from math_assistant_agent.agent.states import GraphAgentState
from math_assistant_agent.config import (
    GROQ_EXTRACTION_TEMPERATURE,
    GROQ_MODEL_NAME,
    KEYWORD_EXTRACTION_PROMPT,
)


def retrieve_key_words(
    client,
    system_prompt: str = KEYWORD_EXTRACTION_PROMPT,
    state: GraphAgentState = None,
    provider_name: str = "groq",
    model: str = GROQ_MODEL_NAME,
    temperature: float = GROQ_EXTRACTION_TEMPERATURE,
) -> dict:
    """Read user question and return key words"""
    question = state.question
    print(f"Extracting key words from {question}")

    content = llm_response(
        client,
        provider_name=provider_name,
        system_prompt=system_prompt,
        question=question,
        model=model,
        temperature=temperature,
        json_mode=True,
    )
    try:
        result_json = json.loads(content)

        extracted_tags = result_json.get("key_words", [])
        print(f"Extracted tags: {extracted_tags}")

        return {"key_words": extracted_tags}
    except Exception as error:
        print(f"Error: {error}")
        return {"key_words": []}
