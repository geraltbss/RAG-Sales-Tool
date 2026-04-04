# RAG-Based Sales Data Analysis - Demo Guide

## System Overview

This system uses Retrieval-Augmented Generation (RAG) to answer analytical questions about a Superstore retail dataset containing 9,994 transactions from 2014–2017. Instead of feeding the entire dataset to an LLM, we convert the data into searchable text documents, store them in a vector database, and retrieve only the relevant pieces when a question is asked.

### Tech Stack

| Component            | Choice                    | Why                                                    |
|----------------------|---------------------------|--------------------------------------------------------|
| Language             | Python 3.13.12            | Course requirement                                     |
| Vector Database      | ChromaDB                  | Simple, embedded, no server setup needed               |
| Embedding Model      | all-MiniLM-L6-v2          | Fast, lightweight (90MB), good quality for short texts |
| LLM                  | Mistral 7B via Ollama     | Runs locally on 6GB VRAM, strong analytical reasoning  |
| UI                   | Gradio                    | Clean web interface, minimal code                      |

---

## Pipeline Architecture

```
                         ┌──────────────────────┐
                         │  Sample-Superstore.csv │
                         │   (9,994 transactions) │
                         └──────────┬───────────┘
                                    │
                         ┌──────────▼───────────┐
                         │  data_preparation.py   │
                         │  • Transaction texts    │
                         │  • Aggregated summaries │
                         │  • Statistical analyses │
                         └──────────┬───────────┘
                                    │
                              5,168 text chunks
                                    │
                         ┌──────────▼───────────┐
                         │   vector_store.py      │
                         │  • Embed with MiniLM   │
                         │  • Store in ChromaDB   │
                         │  • Cosine similarity    │
                         └──────────┬───────────┘
                                    │
              User Question ────────┤
                                    │
                         ┌──────────▼───────────┐
                         │   rag_pipeline.py      │
                         │  1. Classify query      │
                         │  2. Hybrid retrieval    │
                         │  3. Build prompt        │
                         │  4. Mistral generates   │
                         └──────────┬───────────┘
                                    │
                              Answer with data
```

---

## Part 1: Data Preparation (data_preparation.py)

The raw CSV has 21 columns per row. An LLM can't reason over 9,994 raw rows, so we convert the data into three types of natural language documents:

### 1. Transaction Descriptions (5,009 documents)
Each unique order is converted into a readable paragraph. Orders with multiple items are grouped together. Example:

> "Order CA-2016-152156 placed on November 08, 2016 by Claire Gute (Consumer segment) from Henderson, Kentucky (South region). Shipped via Second Class. Items (2): Bush Somerset Collection Bookcase: $261.96, qty 2... Order total: $993.90, total profit: $261.50, profit margin: 26.3%"

### 2. Aggregated Summaries (141 documents)
Pre-computed rollups that the LLM needs for analytical questions:
- **48 monthly summaries** - total sales, profit, orders, avg discount per month
- **16 quarterly summaries** - same metrics per quarter
- **4 yearly summaries** - with year-over-year growth percentages
- **3 category summaries** - Furniture, Office Supplies, Technology
- **17 sub-category summaries** - Bookcases, Chairs, Phones, etc.
- **4 regional summaries** - West, East, Central, South with top states
- **49 state summaries** - per-state breakdown with top cities

### 3. Statistical Summaries (18 documents)
Higher-level analytical documents:
- Dataset overview (total sales, customers, date range)
- Top 10 / Bottom 10 products by sales and profit
- Discount impact analysis (how discounts affect profit margins)
- Segment analysis (Consumer, Corporate, Home Office)
- Shipping mode analysis
- Year-over-year category trends
- Seasonality patterns (average monthly sales)
- Top 10 cities
- Region comparison table

### Chunking Strategy
We don't use arbitrary character-based splitting. Instead, each document is a semantically complete unit (one order, one monthly summary, one category analysis). Average chunk size is 460 characters, max is ~1,950 - all well within the embedding model's 256-token sweet spot.

---

## Part 2: Vector Database (vector_store.py)

### Embedding
All 5,168 documents are embedded using `all-MiniLM-L6-v2` (384-dimensional vectors). This model maps text to a vector space where semantically similar texts are close together.

### Storage
ChromaDB stores the vectors with metadata (document type, year, month, region, category, etc.) using cosine similarity for distance measurement. The database is persisted to disk in `./chroma_db/`.

### Retrieval Functions
- **Basic similarity search** - find the k most similar documents to a query
- **Metadata filtering** - filter by region, category, year, document type
- **Hybrid search** - combines general similarity search with targeted searches across all summary types, deduplicates, and ranks by relevance. This is the main retrieval function used by the pipeline.

### Why Hybrid Search Matters
A simple similarity search for "sales trend over 4 years" might return individual transaction descriptions that mention "sales." Hybrid search ensures we also pull in yearly summaries, monthly summaries, and trend statistics - the documents that actually contain the aggregated data needed to answer the question.

---

## Part 3: RAG Pipeline (rag_pipeline.py)

### Step 1: Query Classification
Each question is classified into categories (trend, seasonality, category, regional, comparative, profitability) using keyword matching. This determines which extra document types to boost during retrieval.

### Step 2: Smart Retrieval
Based on the classification:
- Trend questions -> extra yearly, monthly, quarterly summaries
- Category questions -> extra category and sub-category summaries
- Regional questions -> extra regional and state summaries
- General questions -> standard hybrid search

### Step 3: Prompt Engineering
The retrieved context (up to 6,000 characters) is injected into a structured prompt with:
- A system role defining the assistant as a sales data analyst
- Instructions to use ONLY the provided data, be specific with numbers
- The retrieved data clearly delimited
- The user's question

### Step 4: Generation
Mistral 7B generates the answer with temperature=0.1 (low randomness for factual accuracy). The model runs locally via Ollama, so no data leaves the machine.

---

## Part 4: Analysis Queries (run_analysis.py)

The system handles 8 queries across all required categories:

| Category    | Example Query                                                |
|-------------|--------------------------------------------------------------|
| Trend       | What is the sales trend over the 4-year period?              |
| Trend       | Which months show the highest sales? Is there seasonality?   |
| Category    | Which product category generates the most revenue?           |
| Category    | What sub-categories have the highest profit margins?         |
| Regional    | Which region has the best sales performance?                 |
| Regional    | Which cities are the top performers?                         |
| Comparative | Compare Technology vs. Furniture sales trends.               |
| Comparative | How does the West compare to the East in profit?             |

---

## File Summary

| File                  | Lines | Purpose                                          |
|-----------------------|-------|--------------------------------------------------|
| data_preparation.py   | ~300  | CSV -> 5,168 text chunks with metadata            |
| vector_store.py       | ~250  | ChromaDB embedding, storage, retrieval functions  |
| rag_pipeline.py       | ~220  | Query classification, prompting, LLM integration  |
| run_analysis.py       | ~100  | Runs 8 demo queries, saves results to file        |
| app.py                | ~170  | Gradio web UI for interactive demos               |
| requirements.txt      | -     | Python dependencies                               |
| README.md             | -     | Setup instructions                                |

---

## Key Design Decisions

1. **Semantic chunking over character splitting** - each chunk is a complete logical unit, not an arbitrary 1000-char slice. This gives the embedding model better representations and the LLM cleaner context.

2. **Pre-computed aggregations** - the most important decision. Without monthly/yearly/category summaries, the LLM would have to somehow aggregate thousands of individual transactions to answer "what's the sales trend?" That doesn't work. The summaries make analytical queries possible.

3. **Hybrid retrieval with query-aware boosting** - instead of one-size-fits-all retrieval, the system detects what kind of question is being asked and pulls extra relevant summary types.

4. **Low temperature generation** - 0.1 temperature keeps the LLM grounded in the provided data rather than hallucinating numbers.

5. **Local-only stack** - everything runs on the local machine (Ollama + ChromaDB). No API keys, no cloud costs, no data leaving the machine.
