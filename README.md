# 🧠 AI Math Assistant Agent

An advanced, autonomous Mathematical Assistant Agent built on top of the `Qwen3-4B-Thinking-2507` large language model. This project combines Supervised Fine-Tuning (QLoRA) using real-world solved mathematical problems, Graph-based Retrieval-Augmented Generation (RAG), and agentic orchestration via LangChain/LangGraph.

## 🚀 Project Overview

The goal of this project is to create an AI agent capable of deep mathematical reasoning. Instead of relying purely on a base LLM, this agent is fine-tuned on high-quality, community-vetted mathematical discussions and is orchestrated to use external tools (like Graph Databases and Python calculators) to ensure rigorous and accurate step-by-step problem-solving.

### 🛠️ Tech Stack
* **Base Model:** `Qwen/Qwen3-4B-Thinking-2507` (4B parameters, native 262k context length)
* **Data Sourcing:** StackExchange API (MathOverflow / Mathematics)
* **Fine-Tuning:** HuggingFace `transformers`, `peft` (LoRA), `bitsandbytes` (4-bit quantization), `trl` (SFTTrainer)
* **Orchestration:** LangChain, LangGraph (`create_react_agent`)
* **Serving/Inference:** vLLM / Ollama
* **Long-term Memory (Planned):** Neo4j (Graph Database)
* **Short-term Memory:** LangGraph Checkpointer (`InMemorySaver`) with custom Thinking-Tag filters.

---

## 🗺️ Roadmap & Implementation Phases

### Phase 1: Data Acquisition & Preprocessing (✅ Completed)
* **API Extraction:** Harvested highly-voted, accepted answers from the Math StackExchange and MathOverflow using the `/search/advanced` API endpoint.
* **HTML Parsing:** Processed raw API responses using `BeautifulSoup` to strip unnecessary HTML while strictly preserving MathJax/LaTeX formulas.
* **Dataset Formatting:** Structured the cleaned data into the `ShareGPT/Messages` JSONL format, aligning with the `system`, `user`, and `assistant` roles required by the Qwen3 chat template.

### Phase 2: Supervised Fine-Tuning (QLoRA) (✅ Completed)
* **Quantization:** Loaded the 4B parameter model in 4-bit precision to fit within local GPU VRAM limits.
* **Adapter Training:** Applied LoRA to the attention matrices (q_proj, k_proj, v_proj, etc.) for efficient parameter updates.
* **Hardware Optimization:** Fixed multi-GPU `DataParallel` conflicts by forcing `device_map={"": 0}` during the `trl` SFTTrainer execution.

### Phase 3: Model Merging & Deployment (🚧 In Progress)
* **Weight Merging:** Unloaded the 4-bit quantization, loaded the base model in `bfloat16`, and merged it with the trained LoRA adapters (`merge_and_unload`) to create a standalone model.
* **Inference Engine:** Deploying the merged model using high-performance C++ engines like **vLLM** or **Ollama** to expose an OpenAI-compatible API.
* **Optimized Parameters:** Configured the generation settings according to Qwen's official recommendations for reasoning tasks: `Temperature=0.6`, `Top-P=0.95`, `Top-K=20`, and `max_new_tokens=32768` (up to `81920` for extreme complex math).

### Phase 4: Agentic Orchestration & Memory (🚧 In Progress)
* **LangGraph Integration:** Orchestrating the conversation loop using `create_react_agent`.
* **The "Thinking" Filter:** Implemented a critical middleware to strip `<think>...</think>` tags from the agent's past responses before saving them to the LangGraph checkpointer. This prevents the degradation of the model's reasoning capabilities in multi-turn conversations.

### Phase 5: Advanced Tooling & Knowledge Graph (🔮 Planned)
* **Graph RAG (Neo4j):** Building a semantic retriever to inject complex mathematical theorems and concepts directly into the prompt from a Graph Database.
* **Tool Calling:** Enabling the agent to write and execute Python code in an isolated sandbox or call the Wolfram Alpha API for deterministic calculations.
* **Semantic Caching:** Implementing Redis to cache identical mathematical calculations to save VRAM and reduce latency.

### Phase 6: Model Distillation (🔮 Planned)
* **Teacher-Student Distillation:** Utilizing the fine-tuned 4B model (or a larger 30B variant) to generate synthetic step-by-step solutions.
* **Cost Optimization:** Fine-tuning a smaller model (e.g., 0.5B or 1.5B parameters) purely on the 4B model's outputs to achieve similar reasoning accuracy at a fraction of the computational cost.

---

## ⚠️ Important Best Practices for Qwen3

If you are contributing or running this project locally, please adhere to the following rules based on the official Qwen3 documentation:
1. **System Prompt Standardization:** Always include the instruction: *"Please reason step by step, and put your final answer within \boxed{}."*
2. **Multi-turn History:** The historical model output passed to the API must **only** include the final output part. Never include the `<think>` content in the conversation history payload.

---

## 🤝 Acknowledgments
* **[StackExchange API](https://api.stackexchange.com/)** for providing the rich mathematical data.
* **[Qwen Team](https://huggingface.co/Qwen)** for releasing the powerful open-weights Qwen3-Thinking models.
* **HuggingFace** for the `trl` and `peft` ecosystems.

---

## 📄 License

This project's code is licensed under the [MIT License](LICENSE).

Content sourced from Math StackExchange / MathOverflow (questions, answers, and anything derived from
them, such as the training dataset and knowledge graph) is licensed by StackExchange under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/), per their
[Terms of Service](https://stackoverflow.com/legal/terms-of-service/public). That license requires
attribution and share-alike for redistributed data — it applies independently of this repo's MIT code
license.