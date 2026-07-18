# 🧠 AI Math Assistant Agent

General-purpose LLMs often produce math answers that *look* right but skip steps, hallucinate intermediate results, or can't justify how they got from problem to answer. This project builds an assistant agent that grounds its reasoning instead of just asserting it — combining supervised fine-tuning on real, community-vetted solved problems, a structured knowledge graph that makes each solution's reasoning steps individually inspectable, and agentic orchestration that lets the model call external tools rather than relying solely on its own weights.

## 🚀 Project Overview

The goal is an agent capable of rigorous, traceable mathematical reasoning, built from three complementary layers:

1. **Fine-tuning on real, vetted data.** Rather than relying purely on a generic pretrained model, a compact open-weight LLM is adapted (QLoRA) on high-quality, accepted-answer discussions sourced from math Q&A communities.
2. **A knowledge graph, not just a black box.** Each question/answer pair is decomposed into a `Domain -> Concept -> ResolutionStep` chain and stored as a graph, so a solution's reasoning can be inspected, queried, and reused step by step — not just regenerated from scratch every time.
3. **Agentic orchestration.** The model is wrapped in an agent loop capable of calling external tools (graph retrieval, calculators/sandboxes) to verify and ground its output, instead of answering purely from memorized weights.

The base LLM is an implementation detail, not the point — the pipeline (data → fine-tuning → graph → agent) is designed to work with whatever compact, reasoning-capable open-weight model is swapped in.

### 🛠️ Tech Stack
* **Data Sourcing:** StackExchange API (MathOverflow / Mathematics)
* **Fine-Tuning:** HuggingFace `transformers`, `peft` (LoRA), `bitsandbytes` (4-bit quantization), `trl` (SFTTrainer)
* **Knowledge Graph:** Backend-agnostic node/edge graph, enriched via structured LLM extraction (Gemini API + Pydantic schemas), visualized with PyVis
* **Orchestration:** LangChain, LangGraph (`create_react_agent`)
* **Serving/Inference:** vLLM / Ollama
* **Long-term Memory (Planned):** Neo4j (Graph Database)
* **Short-term Memory:** LangGraph Checkpointer (`InMemorySaver`) with custom Thinking-Tag filters.

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