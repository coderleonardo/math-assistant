"""Node 3: read the question + retrieved context and generate the final answer."""

from math_assistant_agent.agent.llm import llm_response
from math_assistant_agent.agent.states import GraphAgentState
from math_assistant_agent.config import ANSWER_GENERATION_PROMPT, GROQ_MODEL_NAME


def brain(
    client,
    state: GraphAgentState,
    provider_name: str = "groq",
    model: str = GROQ_MODEL_NAME,
    temperature: float = 0.2,
):
    """Read the user question and the retrieved context and generate the final answer"""
    if isinstance(state, dict):
        question = state.get("question", "")
        graph_context = state.get("graph_context", "")
    else:
        question = state.question
        graph_context = state.graph_context

    print(f"Generating the final answer for question {question[:30]}... (print truncated).")

    # Checagem de segurança se o Nó 2 não encontrou nada
    if not graph_context or "None information" in graph_context:
        answer = "I'm sorry, I couldn't find relevant information about this question in the Knowledge Graph."
        print("KG context is empty.")
        return {"answer": answer}

    system_prompt = ANSWER_GENERATION_PROMPT.format(graph_context=graph_context)

    try:
        # ATENÇÃO: Garanta que a sua função llm_response não esteja forçando
        # JSON mode aqui, ou o LLM vai quebrar!
        result_text = llm_response(
            client=client,
            provider_name=provider_name,
            system_prompt=system_prompt,
            question=question,
            model=model,
            temperature=temperature,
            json_mode=False,
        )

        print("Answer generated successfully.")

        # CORREÇÃO 1: Retornando a variável correta gerada pelo LLM
        return {"answer": result_text}

    except Exception as error:
        print(f"Error generating answer: {error}")
        # CORREÇÃO 2: Retornando um dicionário válido para o LangGraph não quebrar
        return {"answer": "An error occurred while generating the mathematical response."}
