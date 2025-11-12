from sentence_transformers import SentenceTransformer
from ollama import Client
import chromadb, json, re

embedder = SentenceTransformer('all-MiniLM-L6-v2')

client = chromadb.PersistentClient(path="./rice_courses_db")
collection = client.get_or_create_collection("rice_courses")
ollama = Client(host="http://localhost:11434")

EXPANSION_SYSTEM = """\
You are helping improve course search. Convert a student's messy request into structured search hints.
Output valid JSON ONLY with these keys:
- semantic_query: short paraphrase of the user's intent
- expanded_queries: array of 3-6 diverse phrasings capturing synonyms and likely intents
- must_have_keywords: up to 5 essential keywords/phrases
- nice_to_have_keywords: up to 10 optional keywords/phrases
- facet_filters: object with zero or more of:
  { "distribution_group": string,
    "diversity_credit": boolean,
    "department": string,
    "level": string,  // e.g., "100/200/300/400" or "grad/undergrad"
    "program": string // e.g., "ASIA", "HUMA", etc.
  }
Only produce compact JSON—no explanations.
"""

def expand_query_with_llm(raw_query: str) -> dict:
    user_prompt = f"""\
Raw user query:
{raw_query}

Consider Rice University context and common catalog language (e.g., "Distribution Group I/II/III", "Diversity Credit").
Infer relevant synonyms (e.g., "Asian diaspora" ↔ "Asian American studies", "migration", "transnational", "diasporic communities", "ethnic studies").
"""
    response = ollama.generate(
        model="llama3:latest",
        prompt=EXPANSION_SYSTEM + "\n" + user_prompt,
        options={"temperature": 0.2}
    )
    # Ollama returns a dict with 'response' text
    text = response["response"].strip()
    # Keep only the JSON (defensive)
    json_str = re.search(r'\{.*\}', text, flags=re.S).group(0)
    return json.loads(json_str)

from collections import defaultdict
from math import inf

def _facet_to_where(facets: dict):
    if not facets: 
        return None
    where = {"$and": []}
    for k, v in facets.items():
        if v is None: 
            continue
        # Map your schema carefully; adjust keys to your metadata field names
        if k == "diversity_credit":
            where["$and"].append({"diversity_credit": {"$eq": bool(v)}})
        elif k == "distribution_group":
            where["$and"].append({"distribution_group": {"$eq": str(v)}})
        elif k == "department":
            where["$and"].append({"department": {"$eq": str(v)}})
        elif k == "level":
            where["$and"].append({"level": {"$eq": str(v)}})
        elif k == "program":
            where["$and"].append({"program": {"$eq": str(v)}})
    if not where["$and"]:
        return None
    return where

def _encode(q: str):
    return embedder.encode([q], normalize_embeddings=True)[0]

def rrf_fuse(result_lists, k=10, k_rrf=60):
    """
    result_lists: list of lists of (id, score, metadata, document)
    RRF score = sum(1 / (k_rrf + rank_i)) across lists
    """
    ranks = defaultdict(lambda: inf)
    # Build rank positions
    for results in result_lists:
        for rank, (rid, _score, meta, doc) in enumerate(results, start=1):
            ranks[(rid, doc)] = min(ranks[(rid, doc)], rank)

    fused = defaultdict(float)
    payload = {}
    for results in result_lists:
        for rank, (rid, _score, meta, doc) in enumerate(results, start=1):
            key = (rid, doc)
            fused[key] += 1.0 / (k_rrf + rank)
            payload[key] = (meta, _score)

    # sort by fused score
    ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    final = []
    for (rid, doc), fused_score in ordered[:k]:
        meta, orig_score = payload[(rid, doc)]
        final.append({"id": rid, "doc": doc, "meta": meta, "fused_score": fused_score, "orig_score": orig_score})
    return final


def query_chroma_multi(collection, queries, where=None, n_results=20):
    result_lists = []
    for q in queries:
        emb = _encode(q)
        res = collection.query(
            query_embeddings=[emb],
            n_results=n_results,
            where=where
        )
        # unpack the single query results into a flat list
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]  # smaller is closer for Chroma cosine? (check your distance metric)
        # Convert distance to a similarity-like number if you want (optional)
        out = list(zip(ids, dists, metas, docs))
        result_lists.append(out)
    fused = rrf_fuse(result_lists, k=10)
    return fused
def expanded_retrieve(raw_query: str, base_where=None, top_k=10):
    exp = expand_query_with_llm(raw_query)

    # Build candidate queries
    candidate_queries = [raw_query]
    if exp.get("semantic_query"):
        candidate_queries.append(exp["semantic_query"])
    if exp.get("expanded_queries"):
        candidate_queries.extend(exp["expanded_queries"])

    # Optional: prepend “query:” if you switch to an E5 model later
    # candidate_queries = [f"query: {q}" for q in candidate_queries]

    # Merge facet filters with any caller-provided filters
    where_from_facets = base_where
    if base_where and where_from_facets:
        where = {"$and": [base_where, where_from_facets]}
    else:
        where = base_where or where_from_facets

    fused = query_chroma_multi(collection, candidate_queries, where=where, n_results=25)

    # Optional keyword post-filter (strong “must-have” curation)
    

    return exp, fused[:top_k]

base_where = {
    "$and": [
        {"distribution_group": {"$eq": "Distribution Group II"}},
        {"diversity_credit": {"$eq": False}}
    ]
}

expansion, results = expanded_retrieve("Give me a history class that is not about european hstory and also not russia and ukraine", base_where=base_where, top_k=10)

import pandas as pd
df = pd.DataFrame([{
    "id": r["id"],
    "title": r["meta"].get("title"),
    "course": r["meta"].get("course"),
    "department": r["meta"].get("department"),
    "distribution_group": r["meta"].get("distribution_group"),
    "diversity_credit": r["meta"].get("diversity_credit"),
    "fused_score": r["fused_score"]
} for r in results])

print("Expansion JSON:\n", json.dumps(expansion, indent=2))
print("\nTop results:\n", df)