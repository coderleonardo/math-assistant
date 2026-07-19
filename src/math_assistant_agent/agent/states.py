"""Shared state passed between the agent's LangGraph nodes."""

from typing import List

from pydantic import BaseModel, Field


class GraphAgentState(BaseModel):
    question: str = Field(default="", description="User question about some math concept.")
    key_words: List[str] = Field(default_factory=list, description="Tags and concepts retrieved.")
    graph_context: str = Field(default="", description="Raw text retrieved from the Knowlodge Graph.")
    answer: str = Field(default="", description="Model final answer.")
