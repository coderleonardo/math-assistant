import os

from google import genai


def build_gemini_client(api_key=None):
    """Build a google.genai.Client, falling back to the GEMINI_API_KEY env var.

    Raises ValueError if no key is found either way.

    Example:
        client = build_gemini_client()  # or build_gemini_client(api_key="AI...")
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found (not passed as an argument, not set as an environment variable)."
        )
    return genai.Client(api_key=api_key)
