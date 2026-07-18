import requests


def fetch_math_dataset(api_key=None, num_questions=5):
    """Fetch top-voted, accepted-answer questions from the MathOverflow StackExchange API.

    Returns a list of dicts with question_id, answer_id, title, prompt (question body),
    completion (accepted answer body), tags, question_score, view_count, link, and
    answer_score. Skips questions without both an accepted answer and a non-empty body.
    Prints and returns [] on a non-200 response instead of raising.

    Example:
        items = fetch_math_dataset(num_questions=5)
        items[0]["title"]
    """
    # Mudamos para search/advanced para suportar o filtro de 'accepted'
    url = "https://api.stackexchange.com/2.3/search/advanced"

    params = {
        "site": "mathoverflow.net",  # Mantive o MathOverflow, altere para "math" se quiser matemática geral
        "order": "desc",
        "sort": "votes",  # Voltando para 'votes' para priorizar a qualidade
        "accepted": "true",  # O filtro de status vital para o Fine-Tuning
        "pagesize": num_questions,
        "filter": "!6WPIomnMNcVD9",  # Filtro gerado para trazer o corpo das perguntas e respostas
    }

    if api_key:
        params["key"] = api_key

    print(f"Buscando {num_questions} questões estruturadas do Math Stack Exchange...")
    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("❌ Erro na API:", response.json())
        return []

    data = response.json()
    dataset = []

    # Estruturando os pares Prompt/Completion
    for item in data.get("items", []):
        prompt_text = item.get("body", "")
        completion_text = ""
        answer_score = None
        accepted_id = item.get("accepted_answer_id")

        # Busca a resposta correta dentro do array de respostas
        if "answers" in item:
            for ans in item["answers"]:
                if ans.get("answer_id") == accepted_id:
                    completion_text = ans.get("body", "")
                    answer_score = ans.get("score")
                    break

        if prompt_text and completion_text:
            dataset.append(
                {
                    "question_id": item.get("question_id"),
                    "answer_id": accepted_id,
                    "title": item.get("title"),
                    "prompt": prompt_text,
                    "completion": completion_text,
                    "tags": item.get("tags", []),
                    "question_score": item.get("score"),
                    "view_count": item.get("view_count"),
                    "link": item.get("link"),
                    "answer_score": answer_score,
                }
            )

    return dataset
