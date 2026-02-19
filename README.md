# RAG-built-from-scratch-v2
This project implements a fully modular Retrieval-Augmented Generation (RAG) pipeline from scratch.  
It processes PDF documents, generates semantic embeddings, performs cosine similarity-based retrieval, and produces structured answers using a 4-bit quantized Large Language Model (LLM).

The goal of this project is to demonstrate a deep understanding of retrieval-based AI systems by building the entire pipeline manually rather than relying on high-level frameworks.

---

## Architecture Overview
![RAG Architecture](assets/architecture.png)


The system follows a standard RAG architecture:

1. PDF Ingestion (Loader)
2. Text Chunking
3. Embedding Generation
4. Vector Storage
5. Similarity-Based Retrieval
6. Context Injection into LLM
7. Structured Answer Generation

### System Flow

PDF → Loader → Chunker → Embedder → Vector Store  
User Query → Query Embedding → Retriever → LLM → Final Answer



---

## Core Components

### Loader
Extracts raw text from PDF documents.

### Chunker
Splits long text into manageable semantic chunks.

### Embedder
Generates dense vector embeddings using sentence-transformer models.

### VectorStorage
Stores embeddings alongside their corresponding text chunks in Python lists.

### Retriever
Computes cosine similarity between the query embedding and stored vectors to retrieve the top-k most relevant chunks.

### LLM Integration
Uses a 4-bit quantized instruction-tuned LLM to generate structured answers strictly from retrieved context.

---

## Model Details

- Embeddings: Sentence-Transformers
- LLM: Llama 3.2 3B Instruct (4-bit quantized)
- Quantization: BitsAndBytes (bnb 4-bit)
- Framework: Hugging Face Transformers
- Backend: PyTorch

The LLM is loaded in 4-bit precision to reduce memory usage and enable efficient inference on limited GPU hardware.

---

## Retrieval Strategy

- Cosine similarity for semantic search
- Top-k retrieval
- Minimum similarity threshold filtering
- Deterministic generation (low temperature) for factual consistency
- Context-restricted prompting (no external knowledge allowed)

---

## Example Query

**Question:**  
Why were people of Germany angry with the Weimar Republic?

**System Output (Summarized):**
- Economic crisis and agricultural decline
- War guilt and national humiliation
- Financial burden of the Treaty of Versailles
- Political instability and constitutional weaknesses
- Public dissatisfaction with coalition governance

---

## Design Decisions

- Modular architecture for extensibility
- In-memory vector storage for simplicity
- Explicit separation between retrieval and generation
- Low-temperature inference for reliable factual output
- Context-grounded prompting to prevent hallucination

---

## Current Limitations

- No FAISS or ANN indexing (uses in-memory vector search)
- Single-document ingestion
- No conversational memory
- Designed for single-session inference
- Requires GPU for efficient LLM inference

---

## Future Improvements

- FAISS integration for scalable retrieval
- Multi-document support
- Streamlit-based UI
- Retrieval evaluation metrics
- Conversational memory layer
- Deployment-ready API interface

---
