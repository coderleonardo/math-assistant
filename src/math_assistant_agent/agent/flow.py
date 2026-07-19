"""Wire the three nodes into a LangGraph agent: extractor -> retriever -> brain."""

from langgraph.graph import END, START, StateGraph

from math_assistant_agent.agent.extractor import retrieve_key_words
from math_assistant_agent.agent.generator import brain
from math_assistant_agent.agent.retriever import retrieve_graph_context
from math_assistant_agent.agent.states import GraphAgentState
from math_assistant_agent.config import GROQ_MODEL_NAME, KEYWORD_EXTRACTION_PROMPT


def build_agent(client, graph_data: dict):
    """Compile the math-assistant agent bound to an LLM client and a knowledge graph.

    The nodes need more than the state (a client, the graph, prompts), but LangGraph
    only passes the state, so each node is wrapped in a closure capturing those
    dependencies. Returns the compiled graph, ready for `.invoke(GraphAgentState(...))`.

    Example:
        client = get_llm_client(api_key=os.getenv("GROQ_API_KEY"))
        graph_data = load_graph_json("data/graph_math_2026-07-17-16.json")
        agent = build_agent(client, graph_data)
        result = agent.invoke(GraphAgentState(question="How do I integrate x^2?"))
    """

    # ==========================================
    # Wrappers (Langgraph only uses state)
    # ==========================================
    def wrapper_extract_keywords(state: GraphAgentState):
        return retrieve_key_words(
            client=client,
            system_prompt=KEYWORD_EXTRACTION_PROMPT,
            state=state,
            provider_name="groq",
            model=GROQ_MODEL_NAME,
        )

    def wrapper_retrieve_context(state: GraphAgentState):
        return retrieve_graph_context(
            state=state,
            graph_data=graph_data,
        )

    def wrapper_brain(state: GraphAgentState):
        return brain(client=client, state=state, provider_name="groq", model=GROQ_MODEL_NAME)

    # ==========================================
    # Graph Building (StateGraph)
    # ==========================================
    workflow = StateGraph(GraphAgentState)

    workflow.add_node("extractor", wrapper_extract_keywords)
    workflow.add_node("retriever", wrapper_retrieve_context)
    workflow.add_node("brain", wrapper_brain)

    workflow.add_edge(START, "extractor")
    workflow.add_edge("extractor", "retriever")
    workflow.add_edge("retriever", "brain")
    workflow.add_edge("brain", END)

    return workflow.compile()
