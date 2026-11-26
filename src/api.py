"""Ships - GET /search?q=... : hybrid movie search with every component's score exposed.

  curl "localhost:8000/search?q=a+guy+keeps+waking+up+on+the+same+day&k=5"

Each result shows the BM25 score + rank, the dense (embedding) score + rank, the fused RRF
score and the cross-encoder rerank score, so you can see which component found it.
The response also has the latency of each component and the budget.
GET /dataset downloads the indexed dataset (data/movies.parquet).
Interactive docs: http://localhost:8000/docs

Run:  uvicorn api:app --app-dir src
"""

from __future__ import annotations

import math

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse

from baseline import ROOT, load_movies
from hybrid_search import LATENCY_BUDGET_MS, HybridSearch

app = FastAPI(title="remember-2-rewatch")
search = HybridSearch(load_movies())  # loads the indexes and models once (~30 s)
search.search("warm up")


@app.get("/search")
def search_endpoint(q: str = Query(min_length=2, max_length=500, description="what you remember about the movie"),
                    k: int = Query(10, ge=1, le=50, description="number of results"),
                    rerank: bool = Query(True, description="use the cross-encoder reranker")):
    results, ms = search.top(q, k, rerank)
    clean = lambda v: None if isinstance(v, float) and math.isnan(v) else v  # no rerank score -> null
    return {
        "query": q,
        "latency_ms": {name: round(v, 1) for name, v in ms.items()},
        "latency_budget_ms": LATENCY_BUDGET_MS,
        "results": [{
            "rank": int(r["rank"]), "movie_id": r["movie_id"], "title": r["title"],
            "year": None if r["year"] != r["year"] else int(r["year"]),  # missing year -> null
            "scores": {"bm25": round(float(r["bm25"]), 3), "bm25_rank": int(r["bm25_rank"]),
                       "dense": round(float(r["dense"]), 3), "dense_rank": int(r["dense_rank"]),
                       "rrf": round(float(r["rrf"]), 5), "rerank": clean(round(float(r["rerank"]), 3))},
        } for r in results.astype(object).to_dict("records")],
    }


@app.get("/dataset")
def dataset():
    """The indexed public dataset (CMU + Wikipedia + TMDB, merged), as a parquet file."""
    return FileResponse(ROOT / "data" / "movies.parquet", filename="movies.parquet")
