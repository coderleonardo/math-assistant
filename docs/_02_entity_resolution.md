# 📝 ROADMAP v2.0: Entity Resolution & Semantic Deduplication

**Objective:** Transition from a 1:1 raw-to-canonical mapping to an N:1 architecture, reducing graph redundancy and improving the quality of few-shot examples for the LLM by merging semantically identical questions.

## Phase 1: Embedding Generation

Instead of blindly adding every distilled case to the graph, we will generate a numerical representation (vector) of the core mathematical problem.

* **Target String:** Concatenate the `objective` and `math_setup` fields from the `CanonicalProblem` extraction.
* **Model:** Use a lightweight, open-source embedding model that runs locally to avoid API costs (e.g., `sentence-transformers/all-MiniLM-L6-v2` via HuggingFace) or a cheap API alternative.

## Phase 2: Vector Similarity Search

We need to compare the new problem's vector against all existing canonical problems in the graph.

* **Infrastructure:** For a small graph (< 10,000 nodes), standard Cosine Similarity using `numpy` or `scipy` is sufficient. For larger scaling, integrate a lightweight in-memory vector store like **FAISS** (Facebook AI Similarity Search) or **ChromaDB**.
* **Query:** Search the vector database for the top-1 most similar existing `CanonicalProblem`.

## Phase 3: The Threshold Routing Logic

Implement a strict cosine similarity threshold (e.g., `T = 0.92`).

* **Condition A (Match Found - Similarity $\ge$ 0.92):** * **DO NOT** create a new `CanonicalProblem` node.
* **MERGE:** Append the new `raw_question_id` to a list inside the matched node (e.g., `properties: { "source_raw_ids": ["q_123", "q_456"] }`).
* **EDGE MERGE:** Ensure any new `Concepts` or `Theorems` extracted from this variation are linked to the existing Canonical node.


* **Condition B (New Problem - Similarity $<$ 0.92):**
* Create a new `CanonicalProblem` node in the graph and add its vector to the FAISS index.


## Phase 4: Continuous Refinement (Optional LLM Pass)

If a merge happens, the canonical setup might be too specific to the first question.

* Periodically run a "Graph Maintenance" script that takes a `CanonicalProblem` with multiple `source_raw_ids`, reads all of them, and asks the LLM to rewrite the `math_setup` to be slightly more generalized, capturing the essence of all its sources.

## Recommended Tech Stack for v2.0:

* **Embeddings:** `sentence-transformers` (Python)
* **Vector Store:** `faiss-cpu` (Python)
* **Orchestration:** Integrated directly into `graph_builder_cbr.py`