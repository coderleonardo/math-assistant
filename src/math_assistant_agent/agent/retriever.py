"""Node 2: look up the extracted keywords in the knowledge graph and return context."""

from math_assistant_agent.agent.states import GraphAgentState


def retrieve_graph_context(
    state: GraphAgentState,
    graph_data: dict,
) -> dict:
    """Read key words from state, look for the information in the graph and return the context"""
    if isinstance(state, dict):
        key_words = state.get("key_words", [])
    else:
        key_words = state.key_words

    print(f"Looking for tags like {key_words}")

    if not key_words:
        print(f"'key_words' is empty")
        return {"graph_context": ""}

    contexts = []

    for node in graph_data.get("nodes", []):
        if node["label"] == "Question":
            question = node["properties"].get("text", "").lower()
            title = node["properties"].get("title", "").lower()

            match_ = False
            for kw in key_words:
                if kw.lower() in question or kw.lower() in title:
                    match_ = True
                    break

            if match_:
                question_id = node["id"]
                context = ""
                answer_id = None

                for edge in graph_data.get("edges", []):
                    if edge["source"] == question_id and edge["type"] == "HAS_ACCEPTED_ANSWER":
                        answer_id = edge["target"]
                        break

                if answer_id:
                    for answer_node in graph_data.get("nodes", []):
                        if answer_node["id"] == answer_id:
                            context = answer_node["properties"].get("text", "")
                            break

                context_block = f"""
                RELATED PROBLEM:
                    title: {node["properties"].get("title", "")}
                    accepted solution: {context}
                """

                contexts.append(context_block)

    full_context = "\n\n".join(contexts)

    if not full_context:
        full_context = "None information found."
        print(f"KG does not have information for the selected key words: {key_words}")
    else:
        print(f"Some informations founded.")

    return {"graph_context": full_context}
