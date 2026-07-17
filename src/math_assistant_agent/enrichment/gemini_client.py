import os

from google import genai


def build_gemini_client(api_key=None):
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found (not passed as an argument, not set as an environment variable)."
        )
    return genai.Client(api_key=api_key)
