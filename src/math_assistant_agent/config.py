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
