from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from math_assistant_agent.config import AGENT_SYSTEM_PROMPT


def build_math_agent(llm, tools=None, system_prompt=AGENT_SYSTEM_PROMPT, checkpointer=None):
    if tools is None:
        tools = []
    if checkpointer is None:
        checkpointer = InMemorySaver()

    return create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt,
        checkpointer=checkpointer,
    )
