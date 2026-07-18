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
