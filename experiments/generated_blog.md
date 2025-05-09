Okay, here is a blog post summarizing the TigerVector paper, incorporating only the specified figures and adhering to your formatting requirements.

---

## TigerVector: Bringing High-Performance Vector Search to Graph Databases for Advanced RAG

Retrieval-Augmented Generation (RAG) has become a cornerstone for grounding Large Language Models (LLMs) with external data. While traditional RAG often relies on vector databases storing semantic embeddings, this approach can struggle with complex queries that require understanding relationships between data points – a strength of graph databases.

Enter VectorGraphRAG, a promising hybrid approach that combines the power of vector search for semantic similarity with graph traversal for structural context. The paper "TigerVector: Supporting Vector Search in Graph Databases for Advanced RAGs" introduces TigerVector, a novel system that integrates vector search directly into TigerGraph, a distributed graph database. This unified approach aims to overcome the limitations of using separate systems, offering benefits like data consistency, reduced silos, and streamlined hybrid queries.

Integrating high-performance vector search into a graph database is challenging. TigerVector tackles this through several key innovations:

**A Unified Data Model:** TigerVector introduces a new `embedding` attribute type for vertices. This isn't just a list of floats; it explicitly manages crucial metadata like dimensionality, the model used, index type, and similarity metric. This dedicated type facilitates managing different types of embeddings and ensures compatibility during queries.

**Decoupled Storage:** Recognizing that vector embeddings are often much larger than other attributes, TigerVector stores vectors separately in "embedding segments." These segments mirror the vertex partitioning of the graph, ensuring related vector and graph data reside together for efficient processing. This decoupling also optimizes updates and index management.
[Figure 3](/Users/bran/Desktop/AIgnite-Solutions/AIgnite/test/tem/2501.11216_Figure3): Decoupled Storage. Vectors within a vertex segment (left) are stored separately in another embedding segment (right), while keeping the same ids.

**Leveraging MPP Architecture:** Built within TigerGraph's Massively Parallel Processing (MPP) architecture, TigerVector distributes vector data and processing across multiple machines. Vector indexes (currently supporting HNSW) are built per segment, and queries are parallelized, with results merged by a coordinator.
[Figure 5](/Users/bran/Desktop/AIgnite-Solutions/AIgnite/test/tem/2501.11216_Figure5): Distributed Query Processing. The coordinator prepares top-k vector search requests in the send queue and dispatches requests to worker servers. Each worker conducts top-k search locally and sends IDs and distances as results back to the response pool in the coordinator.

**GSQL Integration:** TigerVector integrates vector search into TigerGraph's GSQL query language. This includes adding `VECTOR_DIST` to `ORDER BY...LIMIT` syntax for declarative search and introducing a flexible `VectorSearch()` function. This function allows vector search results to be easily composed with graph query blocks, enabling complex hybrid queries.

**Advanced Hybrid Search:** TigerVector supports powerful query patterns beyond simple vector similarity, including filtered vector search and vector search on graph patterns. The `VectorSearch()` function can accept a vertex set from a graph query as a filter, allowing users to find similar items *within* a specific graph context (e.g., find similar posts written by people Alice knows).
![Figure 6](../test/tem/2501.11216_Figure6.png): Demonstration of Combing Community Detection and Vector Search. The Person vertices are partitioned into three communities, colored green, blue, and yellow. The top-k Posts from each community are colored red.

**Efficient Updates:** TigerVector supports transactional updates to vector data, leveraging TigerGraph's MVCC scheme and employing background vacuum processes to incrementally merge delta records into vector indexes. This ensures updates are atomic and efficient.

**Performance:** The paper presents extensive experiments comparing TigerVector with other graph databases supporting vector search (Neo4j, Amazon Neptune) and a specialized vector database (Milvus).
[Figure 7](/Users/bran/Desktop/AIgnite-Solutions/AIgnite/test/tem/2501.11216_Figure7): Throughput Evaluation on SIFT100M and Deep100M. TigerVector significantly outperforms Neo4j and Amazon Neptune in throughput and recall for vector search.
[Figure 8](/Users/bran/Desktop/AIgnite-Solutions/AIgnite/test/tem/2501.11216_Figure8): Latency Evaluation on SIFT100M and Deep100M. TigerVector shows significantly lower latency compared to Neo4j and Amazon Neptune.
TigerVector demonstrates performance comparable to, and sometimes even higher than, Milvus, a specialized vector database, particularly in throughput.

**Scalability:** Experiments show TigerVector scales effectively with both the number of nodes and dataset size, leveraging its distributed architecture.
[Figure 9](/Users/bran/Desktop/AIgnite-Solutions/AIgnite/test/tem/2501.11216_Figure9): Node Scalability. TigerVector exhibits near-linear throughput gain when scaling the number of machines.
[Figure 10](/Users/bran/Desktop/AIgnite-Solutions/AIgnite/test/tem/2501.11216_Figure10): Data Size Scalability. Throughput decreases roughly proportionally as the dataset size scales by 10x, demonstrating good scalability.

In conclusion, TigerVector represents a significant step towards a unified platform for graph and vector data, enabling powerful hybrid searches essential for advanced RAG applications. Its performance is competitive with specialized vector databases and significantly surpasses other graph databases with vector capabilities. TigerVector was integrated into TigerGraph v4.2, released in December 2024.

---