import json
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "superstore_sales"
DB_PATH = "./chroma_db"


def get_embedding_function():
    # Return the sentence-transformer embedding function for ChromaDB.
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def get_client() -> chromadb.PersistentClient:
    # Get or create a persistent ChromaDB client.
    return chromadb.PersistentClient(path=DB_PATH)


def get_collection(client: chromadb.PersistentClient = None):
    # Get or create the sales data collection.
    if client is None:
        client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )


def load_chunks(filepath: str = "prepared_chunks.json") -> list[dict]:
    # Load prepared chunks from JSON.
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vector_store(chunks_path: str = "prepared_chunks.json", batch_size: int = 200):
    # Load all prepared chunks and insert them into ChromaDB.
    # Uses batching to avoid memory issues with large datasets.
    
    chunks = load_chunks(chunks_path)
    client = get_client()

    # Delete existing collection if it exists
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = get_collection(client)
    print(f"Created collection '{COLLECTION_NAME}' with {EMBEDDING_MODEL} embeddings")
    print(f"Inserting {len(chunks)} documents in batches of {batch_size}")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [f"doc_{i + j}" for j in range(len(batch))]
        documents = [chunk["text"] for chunk in batch]

        # ChromaDB metadata values must be str, int, float, or bool
        metadatas = []
        for chunk in batch:
            meta = {}
            for k, v in chunk["metadata"].items():
                if isinstance(v, list):
                    meta[k] = ", ".join(str(x) for x in v)
                else:
                    meta[k] = v
            metadatas.append(meta)

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  Inserted batch {i // batch_size + 1}: docs {i} to {i + len(batch) - 1}")

    print(f"\nDone. Collection has {collection.count()} documents.")
    return collection

# Retrieval Functions

def search(query: str, n_results: int = 10, where: dict = None, where_document: dict = None) -> dict:
    collection = get_collection()
    kwargs = {
        "query_texts": [query],
        "n_results": n_results,
    }
    if where:
        kwargs["where"] = where
    if where_document:
        kwargs["where_document"] = where_document

    return collection.query(**kwargs)


def search_by_type(query: str, doc_type: str, n_results: int = 10) -> dict:
    # Search within a specific document type.
    return search(query, n_results=n_results, where={"type": doc_type})


def search_by_region(query: str, region: str, n_results: int = 10) -> dict:
    # Search filtered by region.
    return search(query, n_results=n_results, where={"region": region})


def search_by_category(query: str, category: str, n_results: int = 10) -> dict:
    # Search filtered by category.
    return search(query, n_results=n_results, where={"category": category})


def search_by_year(query: str, year: int, n_results: int = 10) -> dict:
    # Search filtered by year.
    return search(query, n_results=n_results, where={"year": year})


def hybrid_search(query: str, n_results: int = 15) -> list[dict]:
    # Smart retrieval that pulls from multiple document types to givethe LLM a well-rounded context for analytical questions.
    # Returns a deduplicated list of {text, metadata, distance} dicts.
    seen_ids = set()
    results = []

    # 1. General similarity search (cast a wide net)
    general = search(query, n_results=n_results)

    # 2. Search specifically in summaries
    summary_types = [
        "monthly_summary", "quarterly_summary", "yearly_summary",
        "category_summary", "subcategory_summary",
        "regional_summary", "state_summary", "statistical",
    ]
    summary_results = []
    for stype in summary_types:
        try:
            res = search_by_type(query, stype, n_results=3)
            summary_results.append(res)
        except Exception:
            continue

    # Merge and deduplicate
    def _extract(result_dict):
        extracted = []
        if not result_dict["ids"] or not result_dict["ids"][0]:
            return extracted
        for doc_id, doc, meta, dist in zip(
            result_dict["ids"][0],
            result_dict["documents"][0],
            result_dict["metadatas"][0],
            result_dict["distances"][0],
        ):
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                extracted.append({"id": doc_id, "text": doc, "metadata": meta, "distance": dist})
        return extracted

    results.extend(_extract(general))
    for sr in summary_results:
        results.extend(_extract(sr))

    # Sort by relevance
    results.sort(key=lambda x: x["distance"])
    return results


def format_context(results: list[dict], max_chars: int = 6000) -> str:  
    # Format retrieved results into a context string for the LLM prompt.
    # Respects a character budget to avoid exceeding context window.
    context_parts = []
    total_chars = 0

    for r in results:
        entry = f"[{r['metadata'].get('type', 'unknown')}] {r['text']}"
        if total_chars + len(entry) > max_chars:
            break
        context_parts.append(entry)
        total_chars += len(entry)

    return "\n\n".join(context_parts)

# Main

if __name__ == "__main__":
    import time

    print("Building vector store")
    start = time.time()
    collection = build_vector_store()
    elapsed = time.time() - start
    print(f"Build time: {elapsed:.1f}s")

    # Quick test queries
    print("\nTest: similarity search")
    results = search("What are the total sales by region?", n_results=3)
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"  [{meta['type']}] (dist={dist:.4f}) {doc[:120]}")

    print("\nTest: hybrid search")
    hybrid = hybrid_search("Which product category is most profitable?")
    for r in hybrid[:5]:
        print(f"  [{r['metadata']['type']}] (dist={r['distance']:.4f}) {r['text'][:120]}")

    print("\nTest: metadata filter (region=West)")
    results = search_by_region("sales performance", "West", n_results=3)
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        print(f"  [{meta['type']}] {doc[:120]}")
