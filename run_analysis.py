import sys
import time
from rag_pipeline import query, classify_query

QUERIES = [
    # --- Trend Analysis ---
    {
        "category": "Trend Analysis",
        "question": "What is the sales trend over the 4-year period? Has revenue been growing or declining?",
    },
    {
        "category": "Trend Analysis",
        "question": "Which months show the highest sales? Is there a clear seasonality pattern?",
    },
    # --- Category Analysis ---
    {
        "category": "Category Analysis",
        "question": "Which product category generates the most revenue and which has the highest profit margin?",
    },
    {
        "category": "Category Analysis",
        "question": "What sub-categories have the highest profit margins and which ones are losing money?",
    },
    # --- Regional Analysis ---
    {
        "category": "Regional Analysis",
        "question": "Which region has the best sales performance? Compare all four regions.",
    },
    {
        "category": "Regional Analysis",
        "question": "Which cities are the top performers in terms of sales?",
    },
    # --- Comparative Analysis ---
    {
        "category": "Comparative Analysis",
        "question": "Compare Technology vs. Furniture sales and profit trends over the years.",
    },
    {
        "category": "Comparative Analysis",
        "question": "How does the West region compare to the East in terms of profit and profit margin?",
    },
]


def run_all_queries(save_to: str = "analysis_results.txt"):
    # Run all analysis queries and save results.
    output_lines = []

    def log(text=""):
        print(text)
        output_lines.append(text)

    log("=" * 70)
    log("  RAG-Based Sales Data Analysis - Query Results")
    log(f"  Model: Mistral 7B via Ollama")
    log(f"  Embedding: all-MiniLM-L6-v2")
    log(f"  Vector DB: ChromaDB (5,168 documents)")
    log(f"  Date: {time.strftime('%Y-%m-%d %H:%M')}")
    log("=" * 70)

    total_start = time.time()
    current_category = None

    for i, q in enumerate(QUERIES, 1):
        if q["category"] != current_category:
            current_category = q["category"]
            log(f"\n{'#' * 70}")
            log(f"  {current_category.upper()}")
            log(f"{'#' * 70}")

        log(f"\nQuery {i}/{len(QUERIES)}")
        log(f"Q: {q['question']}")
        log("-" * 70)

        try:
            result = query(q["question"], verbose=True)
            log(f"Query types detected: {', '.join(result['query_types'])}")
            log(f"Context length: {result['context_length']} chars")
            log(f"Retrieval time: {result['retrieval_time']}s")
            log(f"Generation time: {result['generation_time']}s")
            log(f"Total time: {result['total_time']}s")
            log("-" * 70)
            log(f"A: {result['answer']}")
        except Exception as e:
            log(f"ERROR: {e}")
            log("Make sure Ollama is running with: ollama serve")
            log("And the model is pulled with: ollama pull mistral")

        log("=" * 70)

    total_time = time.time() - total_start
    log(f"\nTotal execution time: {total_time:.1f}s")
    log(f"Average per query: {total_time / len(QUERIES):.1f}s")

    # Save to file
    with open(save_to, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\nResults saved to {save_to}")


def run_single(question: str):
    # Run a single custom query with detailed output.
    print(f"\nQ: {question}")
    print(f"Query types: {', '.join(classify_query(question))}")
    print("-" * 70)

    result = query(question, verbose=True)

    print(f"\nRetrieved context ({result['context_length']} chars):")
    print("-" * 40)
    # Show first 500 chars of context for debugging
    print(result["context"][:500] + "..." if len(result["context"]) > 500 else result["context"])
    print("-" * 40)
    print(f"\nA: {result['answer']}")
    print(f"\n[Retrieval: {result['retrieval_time']}s | Generation: {result['generation_time']}s | Total: {result['total_time']}s]")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_single(" ".join(sys.argv[1:]))
    else:
        run_all_queries()
