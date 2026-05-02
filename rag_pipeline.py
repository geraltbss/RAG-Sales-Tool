import time
from vector_store import hybrid_search, format_context, search, search_by_type

# Configuration

OLLAMA_MODEL = "mistral"
OLLAMA_BASE_URL = "http://localhost:11434"
TOP_K = 10        
MAX_CONTEXT = 6000 

# Query Classification

QUERY_TYPE_KEYWORDS = {
    "trend": ["trend", "over time", "year-over-year", "growth", "change", "increased", "decreased", "yoy"],
    "seasonality": ["season", "month", "monthly", "peak", "highest month", "lowest month", "quarterly"],
    "category": ["category", "sub-category", "product", "furniture", "technology", "office supplies"],
    "regional": ["region", "state", "city", "west", "east", "south", "central", "geographic"],
    "comparative": ["compare", "versus", "vs", "difference", "better", "worse", "top", "bottom"],
    "profitability": ["profit", "margin", "loss", "profitable", "losing money", "discount"],
}


def classify_query(query: str) -> list[str]:
    # Classify a query into one or more types based on keywords.
    query_lower = query.lower()
    types = []
    for qtype, keywords in QUERY_TYPE_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            types.append(qtype)
    return types if types else ["general"]


# Smart Retrieval

def retrieve_context(query: str, top_k: int = TOP_K, max_chars: int = MAX_CONTEXT) -> str:
    # Retrieve relevant context using hybrid search with query-aware boosting. Prioritizes summary documents for analytical queries.
    query_types = classify_query(query)

    # Start with hybrid search
    results = hybrid_search(query, n_results=top_k)

    # Boost: for specific query types, pull extra targeted docs
    extra = []
    if "trend" in query_types or "seasonality" in query_types:
        extra.extend(_safe_search(query, "yearly_summary", 4))
        extra.extend(_safe_search(query, "monthly_summary", 4))
        extra.extend(_safe_search(query, "quarterly_summary", 3))
        extra.extend(_safe_search(query, "statistical", 3))

    if "category" in query_types:
        extra.extend(_safe_search(query, "category_summary", 3))
        extra.extend(_safe_search(query, "subcategory_summary", 4))

    if "regional" in query_types:
        extra.extend(_safe_search(query, "regional_summary", 3))
        extra.extend(_safe_search(query, "state_summary", 4))

    if "profitability" in query_types:
        extra.extend(_safe_search(query, "statistical", 4))

    # Merge extra results, dedup by id
    seen = {r["id"] for r in results}
    for r in extra:
        if r["id"] not in seen:
            seen.add(r["id"])
            results.append(r)

    # Re-sort by distance
    results.sort(key=lambda x: x["distance"])

    return format_context(results, max_chars=max_chars)


def _safe_search(query: str, doc_type: str, n: int) -> list[dict]:
    # Search by type, returning list of {id, text, metadata, distance} dicts.
    try:
        res = search_by_type(query, doc_type, n_results=n)
        out = []
        if res["ids"] and res["ids"][0]:
            for doc_id, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0]
            ):
                out.append({"id": doc_id, "text": doc, "metadata": meta, "distance": dist})
        return out
    except Exception:
        return []

# Prompt Engineering

SYSTEM_PROMPT = """You are a sales data analyst assistant. You analyze retail sales data from a Superstore dataset spanning 2014-2017 with 9,994 transactions across 3 categories (Furniture, Office Supplies, Technology) and 4 regions (Central, East, South, West).

Your role:
- Answer analytical questions using ONLY the provided context data
- Be specific with numbers, percentages, and comparisons
- If the context doesn't contain enough information, say so clearly
- Structure your answers with clear sections when appropriate
- Highlight key insights and notable patterns

Always base your answers on the data provided in the context. Do not make up numbers."""


def build_prompt(query: str, context: str) -> str:
    # Build the full prompt with system instructions, context, and query.
    return f"""<s>[INST] {SYSTEM_PROMPT}

--- RETRIEVED DATA ---
{context}
--- END DATA ---

Question: {query}

Provide a detailed analytical answer based on the data above. [/INST]"""

# LLM Integration

def get_llm_response(prompt: str) -> str:
    # Send prompt to Ollama and get response.
    try:
        import ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1, "num_ctx": 4096},
        )
        return response["message"]["content"]
    except ImportError:
        # Fallback to requests if ollama package not available
        import requests
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 4096},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]

# Main RAG Query Function

def query(question: str, verbose: bool = False) -> dict:
    t0 = time.time()

    # 1. Classify
    query_types = classify_query(question)

    # 2. Retrieve
    t1 = time.time()
    context = retrieve_context(question)
    retrieval_time = time.time() - t1

    # 3. Build prompt
    prompt = build_prompt(question, context)

    # 4. Generate
    t2 = time.time()
    answer = get_llm_response(prompt)
    generation_time = time.time() - t2

    total_time = time.time() - t0

    result = {
        "question": question,
        "answer": answer,
        "query_types": query_types,
        "context_length": len(context),
        "retrieval_time": round(retrieval_time, 2),
        "generation_time": round(generation_time, 2),
        "total_time": round(total_time, 2),
    }

    if verbose:
        result["context"] = context
        result["prompt"] = prompt

    return result


def print_result(result: dict):
    # Pretty-print a query result.
    print(f"\n{'='*70}")
    print(f"Q: {result['question']}")
    print(f"{'='*70}")
    print(f"Query types: {', '.join(result['query_types'])}")
    print(f"Context: {result['context_length']} chars | "
          f"Retrieval: {result['retrieval_time']}s | "
          f"Generation: {result['generation_time']}s | "
          f"Total: {result['total_time']}s")
    print(f"{'-'*70}")
    print(result["answer"])
    print(f"{'='*70}\n")

# Interactive Mode & Demo

DEMO_QUESTIONS = [
    "What is the sales trend over the 4-year period?",
    "Which months show the highest sales? Is there seasonality?",
    "Which product category generates the most revenue and which is most profitable?",
    "Which region has the best sales performance?",
    "Compare Technology vs. Furniture sales trends over the years.",
]


def run_demo():
    # Run the 5 demo analytical queries.
    print("\n" + "="*70)
    print("  RAG Sales Analysis Demo - 5 Analytical Queries")
    print("="*70)

    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"\nQuery {i}/{len(DEMO_QUESTIONS)}")
        result = query(q)
        print_result(result)


def interactive():
    # Interactive query mode.
    print("\n" + "="*70)
    print("  RAG Sales Data Analysis - Interactive Mode")
    print("  Type 'quit' to exit, 'demo' to run demo queries")
    print("="*70)

    while True:
        try:
            question = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "demo":
            run_demo()
            continue

        result = query(question)
        print_result(result)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    elif len(sys.argv) > 1:
        # Single question from command line
        question = " ".join(sys.argv[1:])
        result = query(question)
        print_result(result)
    else:
        interactive()
