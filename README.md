# CortexDB

A high-performance, 100% local, AI-native vector database and RAG engine built entirely in Go. 

This engine does not use standard API wrappers. Instead, it has its own low-level storage mechanics and spatial mathematics. This design allows for private, offline semantic search over thousands of documents without sending any data to external cloud providers.

## Key Features

* **Zero-Cloud Privacy:** This works directly with local LLMs through Ollama for embedding and generation. Your data stays on your machine.
* **Custom HNSW Graph Index:** This uses a Hierarchical Navigable Small World (HNSW) graph for very fast `O(log N)` semantic search routing. 
* **LSM-Inspired Storage Engine:** Combines an in-memory Memtable with a Write-Ahead Log (WAL) to ensures data durability and fast O(1) exact-match lookups.
* **End-to-End RAG Pipeline:** Includes built-in PDF ingestion engine, smart text chunking, and a chat interface to immediately start talking to your documents.

## System Architecture

This database is built on three core pillars:

1. **The Ingestion Engine:** Parses raw PDFs, chunks the text, and calls a local embedding model (`nomic-embed-text`) to translate human concepts into dense mathematical vectors (`[]float32`).
2. **The Storage Layer (KV & WAL):** The raw text and metadata are safely committed to a Write-Ahead Log on disk and stored in an active memory map for instant retrieval.
3. **The Spatial Index (HNSW):** The generated vectors are inserted into a multi-layered proximity graph. During a search, the engine drops through the layers, mathematically navigating to the closest semantic neighbors using Cosine Similarity.

## 🛠️ Getting Started

### Prerequisites
* [Go 1.22+](https://go.dev/dl/)
* [Ollama](https://ollama.com/) running locally.
* You must pull the required models before running:
  ```bash
  ollama run nomic-embed-text
  ollama run llama3
  ```

## Installation
### 1. Clone the Repo:
```bash
git clone https://github.com/devmanishoffl/CortecDB
cd CortexDB
```

### 2. Install the required PDF parsing dependency:
```bash
go get github.com/ledongthuc/pdf
```

### 3. Run the database server:
```bash
go run .
```

### 4. Open your browser and navigate to http://localhost:8080.

## Usage
**Upload**: Use the Web UI to upload any complex PDF (contracts, documentation, manuals). The server will automatically chunk the text, embed it, and wire it into the HNSW graph.

**Search & Chat**: Ask the system a natural language question. The engine will perform a vector search, retrieve the most relevant paragraphs, and synthesize a cited answer using the local LLM.
