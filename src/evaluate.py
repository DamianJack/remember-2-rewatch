"""Step 4 - Evaluate: run every method on the same test queries and compare.

Test sets (eval/):
  queries.csv        40 memory-style descriptions of well-known films, plot only
  queries_hints.csv  the same 40 + an extra fact, e.g. "90s comedy with robin williams"
Methods: tfidf, bm25, prompt (baselines); dense, hybrid (bm25 + dense, RRF), hybrid_rerank.
Metrics (rank = position of the correct movie):
  Hit@k        share of queries with the correct movie in the top k
  MRR          mean of 1/rank (1.0 = always first)
  p50/p95 ms   latency per query (median / 95th percentile)
It also shows, query by query, where each component wins (eval/component_wins_<set>.csv)
and the latency of every hybrid component against the budget.

Run:  python src/evaluate.py
      python src/evaluate.py --methods bm25 dense hybrid --sets queries
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from baseline import ROOT, BM25Search, PromptSearch, TfidfSearch, load_movies
from hybrid_search import LATENCY_BUDGET_MS, DenseSearch, HybridSearch

METHODS = ["tfidf", "bm25", "prompt", "dense", "hybrid", "hybrid_rerank"]


def component_wins(r: pd.DataFrame) -> pd.DataFrame:
    """Which component helped on which query (lower rank = better)."""
    wins = {
        "bm25 beats dense": r["bm25"] < r["dense"],
        "dense beats bm25": r["dense"] < r["bm25"],
        "fusion beats both": r["hybrid"] < r[["bm25", "dense"]].min(axis=1),
        "rerank improves fusion": r["hybrid_rerank"] < r["hybrid"],
    }
    rows = []
    for name, mask in wins.items():
        rows.append({"component": name, "queries": int(mask.sum()),
                     "examples": "; ".join(f"{t} ({a}->{b})" for t, a, b in
                                           r.loc[mask, ["title", *pair(name)]].head(4).values)})
    return pd.DataFrame(rows).set_index("component")


def pair(name: str) -> list[str]:
    """the two rank columns compared, (loser, winner), for the example text"""
    return {"bm25 beats dense": ["dense", "bm25"], "dense beats bm25": ["bm25", "dense"],
            "fusion beats both": ["bm25", "hybrid"], "rerank improves fusion": ["hybrid", "hybrid_rerank"]}[name]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    ap.add_argument("--sets", nargs="+", default=["queries", "queries_hints"])
    args = ap.parse_args()

    movies = load_movies()
    ids = movies["movie_id"].to_numpy()
    bm25 = BM25Search(movies)
    needs_dense = any(m in args.methods for m in ("dense", "hybrid", "hybrid_rerank"))
    dense = DenseSearch(movies) if needs_dense else None
    hybrid = HybridSearch(movies, bm25, dense) if needs_dense else None
    build = {"tfidf": lambda: TfidfSearch(movies).rank, "bm25": lambda: bm25.rank,
             "prompt": lambda: PromptSearch(movies).rank, "dense": lambda: dense.rank,
             "hybrid": lambda: hybrid.rank, "hybrid_rerank": lambda: hybrid.rank_rerank}
    methods = {m: build[m]() for m in args.methods}
    if hybrid:
        hybrid.search("warm up")  # load the models onto the device before timing

    for name in args.sets:
        queries = pd.read_csv(ROOT / "eval" / f"{name}.csv")
        answers = queries["movie_ids"].str.split("|")
        results, rows = queries[["title", "query"]].copy(), []
        for method, rank in methods.items():
            ranks, ms = [], []
            for q, ans in zip(queries["query"], answers):
                t = time.perf_counter()
                order = rank(q)
                ms.append((time.perf_counter() - t) * 1000)
                hits = np.flatnonzero(np.isin(ids[order], ans))
                ranks.append(hits[0] + 1 if len(hits) else len(ids))  # not found = last place
            r = results[method] = pd.Series(ranks)
            rows.append({"method": method, "Hit@1": (r <= 1).mean(), "Hit@5": (r <= 5).mean(),
                         "Hit@10": (r <= 10).mean(), "MRR": (1 / r).mean(), "median_rank": r.median(),
                         "p50_ms": np.percentile(ms, 50), "p95_ms": np.percentile(ms, 95)})
        results.to_csv(ROOT / "eval" / f"results_{name}.csv", index=False)
        print(f"\n{name}: {len(queries)} queries, {len(movies):,} movies")
        print(pd.DataFrame(rows).set_index("method").round(3).to_string())

        if {"bm25", "dense", "hybrid", "hybrid_rerank"} <= set(methods):
            wins = component_wins(results)
            wins.to_csv(ROOT / "eval" / f"component_wins_{name}.csv")
            print("\nwhere each component wins (rank before -> after):")
            print(wins.to_string())

    if hybrid:  # latency of every component, over all queries of all sets
        texts = pd.concat([pd.read_csv(ROOT / "eval" / f"{s}.csv")["query"] for s in args.sets])
        lat = pd.DataFrame([hybrid.search(q)[2] for q in texts])
        table = lat.describe(percentiles=[.5, .95]).loc[["50%", "95%", "max"]].T.round(1)
        print(f"\nlatency per component, ms ({len(texts)} queries; budget: p95 total <= {LATENCY_BUDGET_MS} ms)")
        print(table.to_string())
        p95 = table.loc["total", "95%"]
        print(f"=> p95 total {p95} ms: {'WITHIN' if p95 <= LATENCY_BUDGET_MS else 'OVER'} budget")


if __name__ == "__main__":
    main()
