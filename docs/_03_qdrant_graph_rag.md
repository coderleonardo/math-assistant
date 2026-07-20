# 📝 ROADMAP: [Advanced GraphRAG Architecture (Qdrant + Neo4j)](https://qdrant.tech/documentation/examples/graphrag-qdrant-neo4j/)

**Objective:** Migrate the MVP (currently based on in-memory JSON) to a robust, dual-database infrastructure. **Qdrant** will act as the primary vector search engine (Semantic Search), while **Neo4j** will serve as the complex relationship engine (Knowledge Graph). The official `neo4j-graphrag-python` library will handle native orchestration between the two.

## Phase 1: Infrastructure Setup and Configuration

Separation of concerns requires configuring both databases.

* **Qdrant (Vector Store):** Create a cluster on Qdrant Cloud or spin up a local Docker container. Create a *Collection* defining the vector dimension (e.g., 1536 for OpenAI, 768 for local models/Ollama).
* **Neo4j (Graph Database):** Set up the Neo4j instance (AuraDB or Local) and secure the credentials.
* **Dependency Integration:** Install the official `neo4j-graphrag[qdrant]` library, which includes native support for the `QdrantNeo4jRetriever`.

## Phase 2: Hybrid Ingestion Pipeline (Extraction and Upsert)

Replace the current build script with a dual-write workflow.

* **LLM-based Extraction:** Continue using Pydantic/Groq to extract `CanonicalProblems`, `Concepts`, and `Theorems` from raw data.
* **Embedding Generation:** Use a model (e.g., `SentenceTransformerEmbeddings`) to vectorize the `math_setup` or `objective` field of each node.
* **Transactional Writing (The Lettria Pattern):**
1. Initiate a transaction in Neo4j to insert the nodes and edges (relationships). 2. Perform an *Upsert* of the corresponding vector in Qdrant, associated with an ID or payload that references the node in Neo4j.
3. **Consistency:** Ensure that if the write operation to Neo4j fails, the vector in Qdrant is rolled back (preventing "ghost" nodes).

## Phase 3: The Retrieval Flow (QdrantNeo4jRetriever)

Update LangGraph's Node 2 (Retriever) to use coordinated search.

* **Step 3.1 - Initial Semantic Search:** When the user's question arrives, it is vectorized. Qdrant searches for the *Top-K* most similar vectors (e.g., the most similar canonical problems).
* **Step 3.2 - Graph Traversal:** The IDs of the vectors retrieved by Qdrant are sent to Neo4j. Neo4j uses these nodes as "entry points" and executes a Cypher query to retrieve immediate neighbors (the necessary `Theorems` and related `Concepts`).
* **Step 3.3 - Context Injection:** The complete subgraphs retrieved from Neo4j are formatted and injected into the prompt for Node 3 (Brain/Generator).

## Phase 4: Optimizations and Graph Maintenance

* **Metadata Filtering (Payload):** Leverage Qdrant's capability for precise filtering. Example: semantically search only within problems tagged "Complex Analysis" (stored as payload in Qdrant).
* **Continuous Entity Resolution:** Before ingesting a new node, query Qdrant to check if a problem with >95% similarity already exists. If so, merge the edges in Neo4j instead of creating duplicates. ---

**Stack Recommendation for the Transition:**

* **Data Orchestrator:** `neo4j-graphrag-python`
* **Embeddings:** `fastembed` or `sentence-transformers`
* **Databases:** `neo4j` (Python driver) and `qdrant-client`