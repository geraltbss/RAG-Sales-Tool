# RAG-Based Sales Data Analysis

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system for analyzing the Superstore retail sales dataset. Users can ask analytical questions in natural language and receive data-grounded answers powered by a local LLM. The system converts raw tabular data into semantically meaningful text chunks, stores them in a vector database, and retrieves relevant context to generate accurate analytical responses.

## Objectives

- Build an end-to-end RAG pipeline bridging traditional data analysis with AI-powered analytics
- Preprocess and chunk tabular sales data into effective text representations for retrieval
- Set up a vector database with embeddings and metadata-filtered retrieval
- Integrate a locally-running LLM for data-grounded analytical response generation
- Evaluate the system across trend, category, regional, and comparative analysis queries

## Dataset

**Superstore Sales Dataset** from Kaggle: [Download Link](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

| Attribute | Detail |
|-----------|--------|
| Transactions | 9,994 |
| Time span | January 2014 – December 2017 |
| Categories | Furniture, Office Supplies, Technology |
| Regions | Central, East, South, West |
| Key fields | Order Date, Customer, Segment, Product, Sales, Quantity, Discount, Profit, Region, State, City |

## Architecture

```
Superstore CSV (9,994 rows)
    │
    ▼
data_preparation.py
    │  → 5,009 transaction descriptions (grouped by Order ID)
    │  → 141 aggregated summaries (monthly, quarterly, yearly, category, regional)
    │  → 18 statistical summaries (top products, seasonality, discount analysis)
    │
    ▼
vector_store.py
    │  → Embed 5,168 documents with all-MiniLM-L6-v2 (384-dim vectors)
    │  → Store in ChromaDB with metadata (type, region, year, category)
    │
    ▼
rag_pipeline.py
    │  → Classify query type (trend, category, regional, comparative)
    │  → Hybrid retrieval: similarity search + targeted summary boosting
    │  → Construct prompt with retrieved context
    │  → Generate response via Mistral 7B (Ollama, temp=0.1)
    │
    ▼
app.py → Gradio web UI (chat interface, example queries, performance metrics)
```

## Tools and Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.9+ | Core implementation |
| Vector Database | ChromaDB | Embedded persistent vector storage with metadata filtering |
| Embedding Model | all-MiniLM-L6-v2 | 384-dim sentence embeddings (90MB, fast, good quality) |
| LLM | Mistral 7B | Local analytical response generation via Ollama |
| LLM Runtime | Ollama | Local model serving, no API costs |
| Data Processing | Pandas, NumPy | CSV loading, aggregation, statistical computation |
| Web UI | Gradio | Interactive chat-based query interface |

## Setup Instructions

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama
Download from https://ollama.ai, then:
```bash
ollama pull mistral
```

### 3. Run the pipeline

**Step 1: Prepare data** (converts CSV to text chunks)
```bash
python data_preparation.py
```

**Step 2: Build vector store** (embeds chunks into ChromaDB)
```bash
python vector_store.py
```

**Step 3: Query the system**
```bash
# Web UI (recommended for demo)
python app.py

# Or: interactive CLI mode
python rag_pipeline.py

# Or: run all 8 analysis queries and save results
python run_analysis.py

# Or: single question
python rag_pipeline.py "Which region has the best sales?"
```

## Project Files

| File | Purpose |
|------|---------|
| `data_preparation.py` | CSV loading, text chunk generation, aggregated summaries |
| `vector_store.py` | ChromaDB setup, embedding, retrieval functions |
| `rag_pipeline.py` | Query classification, prompt engineering, LLM integration |
| `run_analysis.py` | Runs 8 demo queries, saves results to file |
| `app.py` | Gradio web UI |
| `requirements.txt` | Python dependencies |
| `Sample - Superstore.csv` | Source dataset |
