import os

from groq import Groq


def build_groq_client(api_key=None):
    """Build a groq.Groq client, falling back to the GROQ_API_KEY env var.

    Raises ValueError if no key is found either way.

    Example:
        client = build_groq_client()  # or build_groq_client(api_key="gsk_...")
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found (not passed as an argument, not set as an environment variable)."
        )
    return Groq(api_key=api_key)
