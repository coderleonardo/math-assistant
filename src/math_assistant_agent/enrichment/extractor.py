import json

from google.genai import types

from math_assistant_agent.config import GEMINI_EXTRACTION_TEMPERATURE, GEMINI_MODEL_NAME
from math_assistant_agent.enrichment.schemas import GraphExtraction


def extract_graph_entities(client, question_title, question_text, answer_text):
    """
    Uses Gemini with structured outputs (JSON) to read a question and its accepted
    answer and extract the ontology needed to build a knowledge graph: domain,
    concepts, and the step-by-step resolution (Graph Chain of Thought).
    """
    prompt = f"""
    You are a mathematician specialized in data ontologies.
    Analyze the following math problem and its solution. Extract the information needed
    to build a Knowledge Graph.

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
