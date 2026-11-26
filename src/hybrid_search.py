"""Step 3 - Improvement: hybrid search = BM25 + embeddings, fused, then a cross-encoder reranker.

  1. BM25 (keywords)   finds plots that share the query's exact words: names, rare words.
  2. Dense (meaning)   a sentence-embedding model maps query and plots to vectors; close
                       vectors = similar meaning, even with no shared words
                       ("relives the same day" ~ "time loop"). Model: bge-small-en-v1.5, made
                       for retrieval. Long plots are cut into 150-word chunks; a movie's score
                       is its best chunk.
  3. Fusion (RRF)      Reciprocal Rank Fusion: score = 1/(60 + bm25 rank) + 1/(60 + dense rank).
                       Uses ranks, not raw scores, so the two scales never need matching.
  4. Rerank            a cross-encoder reads (query, the movie's best-matching chunk) TOGETHER for
                       the top 50 fused movies and scores how well they match. Slower per pair,
                       but far more precise. The best chunk (not the start of the plot) matters:
                       what people remember, like the ending, is often late in the plot.

Latency budget: LATENCY_BUDGET_MS (p95 of a whole query on a laptop); evaluate.py measures it.

Try:  python src/hybrid_search.py "a guy keeps waking up on the same day"
      (first run embeds all plots once, ~30 min on CPU; saved to data/embeddings_<model>.npz)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from baseline import ROOT, BM25Search, load_movies

EMBED_MODEL = "BAAI/bge-small-en-v1.5"                 # small (33M), trained for retrieval, 384-dim
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge wants this on queries
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # trained to judge query/passage relevance
INDEX = ROOT / "data" / f"embeddings_{EMBED_MODEL.split('/')[-1]}.npz"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
RRF_K, RERANK_DEPTH, CHUNK_WORDS, MAX_CHUNKS = 60, 50, 150, 4
LATENCY_BUDGET_MS = 500


class DenseSearch:
    def __init__(self, movies: pd.DataFrame):
        # CPU: fast enough for one short query, and the GPU ran out of memory building the index
        self.model = SentenceTransformer(EMBED_MODEL, device="cpu")
        if INDEX.exists() and np.load(INDEX)["starts"].size == len(movies):
            self.emb, self.starts = np.load(INDEX)["emb"], np.load(INDEX)["starts"]
        else:  # embed every plot chunk once (~30 min on CPU) and save
            chunks = [[" ".join(w[i:i + CHUNK_WORDS]) for i in range(0, min(len(w), CHUNK_WORDS * MAX_CHUNKS), CHUNK_WORDS)]
                      or [""] for w in movies["plot"].str.split()]
            self.starts = np.cumsum([0] + [len(c) for c in chunks[:-1]])  # first chunk of each movie
            self.emb = self.model.encode([c for cs in chunks for c in cs], batch_size=128, show_progress_bar=True,
                                         normalize_embeddings=True).astype(np.float16)
            np.savez(INDEX, emb=self.emb, starts=self.starts)
        self.emb = self.emb.astype(np.float32)  # float16 on disk (half the size), float32 to compute (much faster)

    def chunk_scores(self, query: str) -> np.ndarray:
        """similarity of the query to every plot chunk"""
        return self.emb @ self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)

    def scores(self, query: str, chunk_scores: np.ndarray | None = None) -> np.ndarray:
        sims = self.chunk_scores(query) if chunk_scores is None else chunk_scores
        return np.maximum.reduceat(sims, self.starts)  # best chunk per movie

    def best_chunk(self, row: int, sims: np.ndarray) -> int:
        """which chunk (0, 1, 2, ...) of movie `row` matched the query best"""
        end = self.starts[row + 1] if row + 1 < len(self.starts) else len(sims)
        return int(np.argmax(sims[self.starts[row]:end]))

    def rank(self, query: str) -> np.ndarray:
        return np.argsort(-self.scores(query), kind="stable")


def ranks_of(scores: np.ndarray) -> np.ndarray:
    """rank (1 = best) of every movie"""
    r = np.empty(len(scores), dtype=np.int64)
    r[np.argsort(-scores, kind="stable")] = np.arange(1, len(scores) + 1)
    return r


class HybridSearch:
    def __init__(self, movies: pd.DataFrame, bm25: BM25Search | None = None, dense: DenseSearch | None = None):
        self.movies = movies
        self.bm25 = bm25 or BM25Search(movies)
        self.dense = dense or DenseSearch(movies)
        # fixed max length: the GPU caches one compiled graph per input shape, so fewer shapes = no memory growth
        self.reranker = CrossEncoder(RERANK_MODEL, device=DEVICE, max_length=256)
        self.plots = movies["plot"].to_numpy()

    def passage(self, row: int, sims: np.ndarray) -> str:
        """what the reranker reads: the chunk of this movie's plot that matched the query best"""
        i = self.dense.best_chunk(row, sims) * CHUNK_WORDS
        return " ".join(self.plots[row].split()[i:i + CHUNK_WORDS])

    def search(self, query: str, rerank: bool = True) -> tuple[np.ndarray, dict, dict]:
        """-> (order of all movies, per-component scores, per-component latency in ms)"""
        ms, t = {}, time.perf_counter()
        bm25 = self.bm25.scores(query)
        ms["bm25"], t = (time.perf_counter() - t) * 1000, time.perf_counter()
        sims = self.dense.chunk_scores(query)
        dense = self.dense.scores(query, sims)
        ms["dense"], t = (time.perf_counter() - t) * 1000, time.perf_counter()
        bm25_rank, dense_rank = ranks_of(bm25), ranks_of(dense)
        rrf = 1 / (RRF_K + bm25_rank) + 1 / (RRF_K + dense_rank)
        order = np.argsort(-rrf, kind="stable")
        ms["fusion"], t = (time.perf_counter() - t) * 1000, time.perf_counter()
        rerank_score = np.full(len(order), np.nan, dtype=np.float32)
        if rerank:
            top = order[:RERANK_DEPTH]
            pairs = [(query, self.passage(row, sims)) for row in top]
            rerank_score[top] = self.reranker.predict(pairs, batch_size=RERANK_DEPTH)
            if DEVICE == "mps":
                torch.mps.empty_cache()  # GPU memory is shared with the whole laptop: give it back
            order = np.concatenate([top[np.argsort(-rerank_score[top], kind="stable")], order[RERANK_DEPTH:]])
            ms["rerank"] = (time.perf_counter() - t) * 1000
        ms["total"] = sum(ms.values())
        scores = {"bm25": bm25, "bm25_rank": bm25_rank, "dense": dense, "dense_rank": dense_rank,
                  "rrf": rrf, "rerank": rerank_score}
        return order, scores, ms

    def rank(self, query: str) -> np.ndarray:
        return self.search(query, rerank=False)[0]

    def rank_rerank(self, query: str) -> np.ndarray:
        return self.search(query, rerank=True)[0]

    def top(self, query: str, k: int = 10, rerank: bool = True) -> tuple[pd.DataFrame, dict]:
        """the k best movies with every component's score, for the API / printing"""
        order, s, ms = self.search(query, rerank)
        top = order[:k]
        out = self.movies.iloc[top][["movie_id", "title", "year"]].reset_index(drop=True)
        out.insert(0, "rank", range(1, len(top) + 1))
        for name, values in s.items():
            out[name] = values[top]
        return out, ms


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "a guy keeps waking up on the same day"
    search = HybridSearch(load_movies())
    search.top(query)  # warm-up: the first call loads the models onto the device
    results, ms = search.top(query)
    print(results.round(3).to_string(index=False))
    print("latency ms:", {k: round(v, 1) for k, v in ms.items()}, f"(budget {LATENCY_BUDGET_MS} ms)")
