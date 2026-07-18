import json

from google.genai import types

from math_assistant_agent.config import GEMINI_EXTRACTION_TEMPERATURE, GEMINI_MODEL_NAME
from math_assistant_agent.enrichment.schemas import GraphExtraction


def extract_graph_entities(client, question_title, question_text, answer_text, known_concepts=None):
    """Extract a GraphExtraction-shaped dict (concepts/resolution_steps) from a Q&A pair.

    Uses Gemini's structured-output mode (response_schema=GraphExtraction), so the
    result always matches the schema exactly. On any error, prints it and returns None
    instead of raising, so a batch loop can skip a failed item.

    known_concepts, if given, is a list of concept names already in the graph — the model
    is asked to reuse an exact existing name when one fits, so the same idea doesn't get a
    new phrasing each time (controlled vocabulary).

    Example:
        client = build_gemini_client()
        extracted = extract_graph_entities(client, title, question_text, answer_text)
    """
    vocabulary_clause = ""
    if known_concepts:
        vocabulary_clause = (
            "\n    The knowledge graph already uses the following concept names. When a "
            "concept in this problem matches one of them, reuse the EXACT existing name "
            "instead of inventing a new phrasing; only introduce a new name when none of "
            f"these accurately fit: {', '.join(known_concepts)}\n"
        )

    prompt = f"""
    You are a mathematician specialized in data ontologies.
    Analyze the following math problem and its solution. Extract the information needed
    to build a Knowledge Graph.
    {vocabulary_clause}
    QUESTION TITLE: {question_title}
    QUESTION TEXT: {question_text}
    ACCEPTED ANSWER: {answer_text}
    """

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=GraphExtraction,
                temperature=GEMINI_EXTRACTION_TEMPERATURE,
            ),
        )

        return json.loads(response.text)

    except Exception as e:
        print(f"Error extracting data with Gemini: {e}")
        return None
