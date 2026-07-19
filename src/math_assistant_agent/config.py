MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"

# Token id for the closing </think> tag, used to split reasoning from the
# final answer in generated output (see notebooks/modeling/math-agent-finetuning.ipynb).
THINK_END_TOKEN_ID = 151668

SYSTEM_PROMPT = (
    "Você é um assistente de matemática avançado. Resolva os problemas "
    "passo a passo, fornecendo explicações claras e rigorosas."
)

AGENT_SYSTEM_PROMPT = (
    "Você é um assistente de matemática avançado. Resolva os problemas passo a passo, "
    "fornecendo explicações claras e rigorosas baseadas nos seus conhecimentos avançados.\n\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)

GENERATION_DEFAULTS = {
    "max_new_tokens": 4096,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}

GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_EXTRACTION_TEMPERATURE = 0.1

GROQ_MODEL_NAME = "openai/gpt-oss-20b"
GROQ_EXTRACTION_TEMPERATURE = 0.1

# Prompt for the agent's keyword-extraction node (agent.extractor.retrieve_key_words).
# Must ask for English keywords so they match the (English) knowledge-graph text.
KEYWORD_EXTRACTION_PROMPT = """You are a mathematical librarian.
Analyze the user's question and extract 1 to 3 mathematical concepts or keywords.
You MUST return ONLY a JSON object with the key "key_words" containing a list of strings (in English, to match the database).
Example: {"key_words": ["polynomials", "roots", "unit circle"]}
"""

# Prompt template for the agent's answer-generation node (agent.generator.brain).
# It is a template: brain() fills {graph_context} with the retrieved knowledge-graph text.
ANSWER_GENERATION_PROMPT = """
    You are an expert Math Teaching Assistant.
    Your goal is to answer the user's question clearly, using ONLY the context provided below.
    If the context contains LaTeX math, format your response using proper LaTeX delimiters ($ for inline, $$ for block).
    Break your explanation into logical steps. Do not hallucinate math theorems outside the context.

    CONTEXT RETRIEVED FROM KNOWLEDGE GRAPH:
    {graph_context}
    """

# StackExchange tags that describe a question's *form* rather than its mathematics.
# They are the graph's biggest hubs by degree, which makes the topic structure hard to
# read: everything routes through "big-list" instead of through a field of mathematics.
# Excluded from the Tag layer by default (see data.graph.build_graph_records).
# A plain blocklist on purpose — MathOverflow's "xx." prefix can't be used to infer this,
# since e.g. ho.history-overview is a legitimate topic tag.
METADATA_TAGS = frozenset(
    {
        "advice",
        "big-list",
        "big-picture",
        "career-development",
        "community",
        "examples",
        "intuition",
        "journals",
        "mathematical-philosophy",
        "mathematics-education",
        "reference-request",
        "soft-question",
        "teaching",
    }
)
