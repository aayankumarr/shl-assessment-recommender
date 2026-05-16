import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import faiss



CATALOG_PATH   = Path("catalog_processed.json")
INDEX_PATH     = Path("faiss_index.bin")
METADATA_PATH  = Path("index_metadata.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BM25_TOP_K      = 20   # candidates from BM25
SEMANTIC_TOP_K  = 20   # candidates from FAISS
RRF_K           = 60   # RRF constant (standard value, don't change)
RERANK_TOP_N    = 10   # final results after cross-encoder



def build_index():
    print("building index from catalog")

    with open(CATALOG_PATH,encoding= "utf-8") as f:
        catalog = json.load(f)

        metadata = []
        search_texts = []

        for item in catalog:
            search_text = item.get("search_text","")
            if not search_text:
                continue
            search_texts.append(search_text)
            metadata.append({
                "name": item["name"],
                "url": item["url"],
                "test_type": item.get("test_type", ""),
                "test_type_labels": item.get("test_type_labels", []),
                "search_text": search_text,
            })

        print(f"Embedding {len(search_texts)} assessment info")

        model = SentenceTransformer(EMBEDDING_MODEL)

        embeddings = model.encode(search_texts,show_progress_bar =True, convert_to_numpy = True)
        
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        faiss.write_index(index, str(INDEX_PATH))

        with open(METADATA_PATH,"w",encoding="utf-8") as f:
            json.dump(metadata, f, indent = 2, ensure_ascii=False)
        print(f"Index built. {len(metadata)} assessments indexed.")





def load_index():
    if not INDEX_PATH.exists() or not METADATA_PATH.exists():
        build_index()
    
    print("Loading index...")
    
    index    = faiss.read_index(str(INDEX_PATH))
    
    with open(METADATA_PATH, encoding="utf-8") as f:
        metadata = json.load(f)
    
    with open(CATALOG_PATH, encoding="utf-8") as f:
        catalog = json.load(f)
    
    search_texts = [item["search_text"] for item in metadata]
    tokenized    = [text.lower().split() for text in search_texts]
    bm25         = BM25Okapi(tokenized)
    
    bi_encoder   = SentenceTransformer(EMBEDDING_MODEL)
    cross_encoder = CrossEncoder(RERANKER_MODEL)
    
    print(f"Index loaded. {index.ntotal} assessments ready.")
    
    return index, metadata, bm25, bi_encoder, cross_encoder





def retrieve(query: str, top_k: int = RERANK_TOP_N,
             index=None, metadata=None, bm25=None,
             bi_encoder=None, cross_encoder=None) -> list[dict]:

    # ── BM25 Search ───────────────────────────────────────────────
    query_tokens = query.lower().split()
    bm25_scores  = bm25.get_scores(query_tokens)
    bm25_top     = np.argsort(bm25_scores)[::-1][:BM25_TOP_K]

    # ── FAISS Search ──────────────────────────────────────────────
    query_vector = bi_encoder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vector)
    _, faiss_indices = index.search(query_vector, SEMANTIC_TOP_K)
    faiss_top = faiss_indices[0]

    # ── RRF Fusion ────────────────────────────────────────────────
    rrf_scores = {}

    for rank, idx in enumerate(faiss_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + RRF_K)

    for rank, idx in enumerate(bm25_top):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (rank + RRF_K)

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    merged = merged[:20]

    # ── Cross-Encoder Reranking ───────────────────────────────────
    pairs = [(query, metadata[idx]["search_text"]) for idx, _ in merged]
    ce_scores = cross_encoder.predict(pairs)

    reranked = sorted(
        zip(ce_scores, [idx for idx, _ in merged]),
        key=lambda x: x[0],
        reverse=True
    )

    # ── Build Results ─────────────────────────────────────────────
    results = []
    for score, idx in reranked[:top_k]:
        item = metadata[idx]
        results.append({
            "name":             item["name"],
            "url":              item["url"],
            "test_type":        item["test_type"],
            "test_type_labels": item["test_type_labels"],
            "score":            float(score),
        })

    return results







if __name__ == "__main__":
    index, metadata, bm25, bi_encoder, cross_encoder = load_index()

    test_queries = [
        "I am hiring a Java developer",
        "I need a personality test for a senior manager",
        "cognitive ability test for graduate recruitment",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        results = retrieve(
            query,
            index=index,
            metadata=metadata,
            bm25=bm25,
            bi_encoder=bi_encoder,
            cross_encoder=cross_encoder,
        )
        for i, r in enumerate(results, 1):
            print(f"  {i:2}. [{r['test_type']}] {r['name']}")
            print(f"      {r['url']}")
