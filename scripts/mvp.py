"""Fast MVP runner for the math-assistant agent.

Loads the knowledge graph, builds the LangGraph agent, and answers a single
question so you can smoke-test the extractor -> retriever -> brain flow end to end.

Usage:
    python scripts/mvp.py
    python scripts/mvp.py "How do I integrate x^2?"
    python scripts/mvp.py "..." --graph data/graph_math_kb_2026-07-18-15.json

Requires GROQ_API_KEY in the environment (or a .env file at the repo root).
"""

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from math_assistant_agent.agent import build_agent, get_llm_client, GraphAgentState
from math_assistant_agent.data import load_graph_json

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GRAPH = REPO_ROOT / "data" / "graph_math_2026-07-17-16.json"
DEFAULT_QUESTION = "Can you explain how we can calculate the integral of x^2?"


def main():
    parser = argparse.ArgumentParser(description="Run the math-assistant agent on one question.")
    parser.add_argument(
        "question",
        nargs="?",
        default=DEFAULT_QUESTION,
        help="Math question to ask the agent.",
    )
    parser.add_argument(
        "--graph",
        default=str(DEFAULT_GRAPH),
        help="Path to the knowledge-graph JSON file.",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set. Add it to your environment or a .env file.")

    client = get_llm_client(api_key=api_key)
    graph_data = load_graph_json(args.graph)
    agent = build_agent(client, graph_data)

    print("\n🚀 STARTING AGENT EXECUTION...")
    result = agent.invoke(GraphAgentState(question=args.question))

    print("\n" + "=" * 50)
    print("🎓 AGENT ANSWER:")
    print("=" * 50)
    print(result["answer"])


if __name__ == "__main__":
    main()
