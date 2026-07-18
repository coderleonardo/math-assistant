import json

import groq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from math_assistant_agent.config import GROQ_EXTRACTION_TEMPERATURE, GROQ_MODEL_NAME
from math_assistant_agent.enrichment.schemas import GraphExtraction


@retry(
    wait=wait_exponential(multiplier=2, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(groq.RateLimitError),
    reraise=True,
)
def extract_graph_entities_groq(
    client, question_title, question_text, answer_text, known_concepts=None
):
    """Extract a GraphExtraction-shaped dict (concepts/resolution_steps) from a Q&A pair.

    Same signature as extract_graph_entities (the Gemini backend) — pass this as
    enrich_graph_records's extract_fn to use Groq's gpt-oss models instead. Retries up
    to 5 times with exponential backoff on groq.RateLimitError; any other error is
    printed and returns None.

    known_concepts, if given, is a list of concept names already in the graph — the model
    is asked to reuse an exact existing name when one fits, so the same idea doesn't get a
    new phrasing each time (controlled vocabulary).

    Example:
        client = build_groq_client()
        extracted = extract_graph_entities_groq(client, title, question_text, answer_text)
    """
    json_schema = GraphExtraction.model_json_schema()

    system_prompt = f"""
    You are a mathematician specialized in data ontologies.
    Extract the information needed to build a Knowledge Graph from the given math problem.
    You MUST respond ONLY with a valid JSON object that strictly follows this schema:
    {json.dumps(json_schema, indent=2)}
    """

    vocabulary_clause = ""
    if known_concepts:
        vocabulary_clause = (
            "\n    The knowledge graph already uses the following concept names. When a "
            "concept in this problem matches one of them, reuse the EXACT existing name "
            "instead of inventing a new phrasing; only introduce a new name when none of "
            f"these accurately fit: {', '.join(known_concepts)}\n"
        )

    user_prompt = f"""
    QUESTION TITLE: {question_title}
    QUESTION TEXT: {question_text}
    ACCEPTED ANSWER: {answer_text}
    {vocabulary_clause}"""

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=GROQ_EXTRACTION_TEMPERATURE,
        )

        data = json.loads(response.choices[0].message.content)
        validated = GraphExtraction(**data)

        return validated.model_dump()

    except groq.RateLimitError:
        raise
    except Exception as e:
        print(f"Error extracting data with Groq: {e}")
        return None
