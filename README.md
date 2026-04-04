# RAG-Based Sales Data Analysis

A Retrieval-Augmented Generation system for analyzing the Superstore sales dataset (2014-2017, 9,994 transactions) using ChromaDB and Mistral 7B via Ollama.

## Setup

### 1. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and start Ollama
Download from https://ollama.ai, then:
```bash
ollama pull llama3.2:3b
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

## Architecture

```
Superstore CSV
    - data_preparation.py (text chunks + aggregated summaries)
    - vector_store.py (ChromaDB + all-MiniLM-L6-v2 embeddings)
    - rag_pipeline.py (hybrid retrieval + Llama 3.2:3B via Ollama)
    - Analytical answers
```

## Files
- 'data_preparation.py' - Data loading, text generation, chunking
- 'vector_store.py' - ChromaDB setup, embedding, retrieval functions
- 'rag_pipeline.py' - RAG pipeline with query classification, prompt engineering, LLM integration
- `requirements.txt` - Python dependencies
- `Sample - Superstore.csv` - Source dataset
