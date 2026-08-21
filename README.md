# Enterprise RAG Knowledge Assistant

An enterprise-style Retrieval-Augmented Generation (RAG) application built using Python, Azure OpenAI and Azure AI Search.

The system allows users to ask questions about enterprise documents and provides grounded answers with source citations.

The project also evaluates the RAG system using a baseline-vs-improved approach covering retrieval quality, answer correctness, groundedness, citation correctness, hallucination rate, latency and token usage.

---

# 1. Project Overview

The objective of this project is to build a production-oriented enterprise knowledge assistant that can answer questions from a controlled document knowledge base.

The application supports:

- PDF documents
- DOCX documents
- XLSX documents
- Document parsing
- Text cleaning
- Chunking with overlap
- Azure OpenAI embeddings
- Azure AI Search
- Vector search
- Hybrid retrieval
- Semantic reranking
- Query decomposition
- Multi-document retrieval
- Evidence sufficiency checking
- Hallucination prevention
- Ambiguous-query handling
- Conversational follow-up questions
- Source citations
- RAG evaluation

---

# 2. Business Problem

Enterprise users often need information from multiple internal documents such as:

- HR policies
- Finance documents
- Sales documents
- IT documentation
- Legal policies
- Pricing documents
- Operational documents

Traditional keyword search may return many documents without understanding the user's intent.

This solution uses RAG to:

1. Retrieve relevant enterprise information.
2. Provide the retrieved information as context to an LLM.
3. Generate an answer grounded in the retrieved context.
4. Provide citations so users can trace the answer back to the source document.

The system is designed to reduce hallucination and improve answer traceability.

---

# 3. High-Level Architecture

```text
                         USER
                           |
                           v
                  +------------------+
                  |   FastAPI API    |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Query Processing |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Query Rewriting  |
                  | / Decomposition  |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Azure AI Search  |
                  |                  |
                  | Vector Search    |
                  | Keyword Search   |
                  | Hybrid Search    |
                  | Semantic Ranking  |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Evidence Check   |
                  +--------+---------+
                           |
                  +--------+---------+
                  |                  |
              Sufficient         Insufficient
                  |                  |
                  v                  v
          +---------------+    +----------------+
          | Azure OpenAI  |    | Safe Response  |
          | GPT Model     |    | No Guessing    |
          +-------+-------+    +----------------+
                  |
                  v
          Grounded Answer
             + Citations