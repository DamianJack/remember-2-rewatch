"""
The full hybrid search in ONE place: BM25 -> dense -> fusion (RRF) -> reranker.
search(question) returns the top 10 movies WITH each part's score (the brief asks for this),
and how long each part took.

Run from the main project folder:
    python search/pipeline.py "a man relives the same day again and again"   # try one question
    python search/pipeline.py --eval                                         # all test questions

To switch the reranker, change RERANKER below (ms-marco first, then bge).
"""
import json
import sys
import time

import bm25s
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, "cleaning")
from common import make_loose_key, make_title_key  # noqa: E402

# ---------- SETTINGS ----------
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MOVIES = "data/merged/movies.json"
DENSE_INDEX = "data/index/dense_bge_small.npy"
DENSE_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TEST_SET = "eval/test_set.json"

CANDIDATES = 100          # each search hands in this many
FUSION_WEIGHTS = (3, 1)   # BM25 : dense
RRF_K = 60
SHORTLIST = 30            # the reranker reads the top 30 of the fused list
RERANK_WORDS = 300
TOP_K = 10


class HybridSearch:
    def __init__(self):
        t0 = time.perf_counter()
        with open(MOVIES, encoding="utf-8") as f:
            self.movies = json.load(f)
        try:
            import torch
            device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        except Exception:
            device = "cpu"

        # BM25
        self.bm25 = bm25s.BM25()
        self.bm25.index(bm25s.tokenize([self._full_text(m) for m in self.movies],
                                       stopwords="en", show_progress=False), show_progress=False)
        # Dense
        self.emb = np.load(DENSE_INDEX).astype(np.float32)
        self.dense_model = SentenceTransformer(DENSE_MODEL, device=device)
        # Reranker
        self.reranker = CrossEncoder(RERANKER, device=device, max_length=512)
        print(f"Ready in {time.perf_counter() - t0:.0f} s  ({len(self.movies)} movies, "
              f"reranker: {RERANKER}, device: {device})")

    # --- texts ---
    @staticmethod
    def _full_text(m):
        parts = [m["title"], *m["other_titles"], *m["genres"], *m["keywords"],
                 m["tagline"], m["tmdb_overview"], m["cmu_plot"], m["wiki_plot"]]
        return " ".join(p for p in parts if p)

    @staticmethod
    def _judge_text(m):
        story = " ".join(p for p in [m["tmdb_overview"], m["wiki_plot"] or m["cmu_plot"]] if p)
        return " ".join(f"{m['title']} ({m['year']}). {story}".split()[:RERANK_WORDS])

    # --- the search ---
    def search(self, question, k=TOP_K):
        times = {}

        t = time.perf_counter()
        docs, scores = self.bm25.retrieve(bm25s.tokenize([question], stopwords="en", show_progress=False),
                                          k=CANDIDATES, show_progress=False)
        bm25 = {int(d): (place + 1, float(s)) for place, (d, s) in enumerate(zip(docs[0], scores[0]))}
        times["bm25_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        q = self.dense_model.encode([QUERY_PREFIX + question], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.emb @ q
        top = np.argpartition(-sims, CANDIDATES)[:CANDIDATES]
        top = top[np.argsort(-sims[top])]
        dense = {int(d): (place + 1, float(sims[d])) for place, d in enumerate(top)}
        times["dense_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        rrf = {}
        for found, weight in [(bm25, FUSION_WEIGHTS[0]), (dense, FUSION_WEIGHTS[1])]:
            for d, (place, _) in found.items():
                rrf[d] = rrf.get(d, 0) + weight / (RRF_K + place)
        shortlist = sorted(rrf, key=rrf.get, reverse=True)[:SHORTLIST]
        times["fusion_ms"] = (time.perf_counter() - t) * 1000

        t = time.perf_counter()
        judge = self.reranker.predict([(question, self._judge_text(self.movies[d])) for d in shortlist],
                                      batch_size=SHORTLIST, show_progress_bar=False)
        order = sorted(range(len(shortlist)), key=lambda i: -judge[i])
        times["rerank_ms"] = (time.perf_counter() - t) * 1000
        times["total_ms"] = sum(times.values())

        results = []
        for i in order[:k]:
            d = shortlist[i]
            m = self.movies[d]
            results.append({
                "index": d, "id": m["id"], "title": m["title"], "year": m["year"],
                "scores": {
                    "bm25": round(bm25[d][1], 3) if d in bm25 else None,
                    "bm25_place": bm25[d][0] if d in bm25 else None,
                    "dense": round(dense[d][1], 3) if d in dense else None,
                    "dense_place": dense[d][0] if d in dense else None,
                    "fusion": round(rrf[d], 4),
                    "fusion_place": shortlist.index(d) + 1,
                    "rerank": round(float(judge[i]), 3),
                },
            })
        return results, {k_: round(v, 1) for k_, v in times.items()}


# ---------- run it ----------
def evaluate(engine):
    by_key = {}
    for i, m in enumerate(engine.movies):
        for title in [m["title"], *m["other_titles"]]:
            for key in {make_title_key(title), make_loose_key(make_title_key(title))}:
                for y in {m["year"], *m["other_years"]}:
                    by_key.setdefault((key, y), set()).add(i)

    def find(title, year):
        key, found = make_title_key(title), set()
        for k_ in (key, make_loose_key(key)):
            for y in (year, year - 1, year + 1):
                found |= by_key.get((k_, y), set())
        return found

    with open(TEST_SET, encoding="utf-8") as f:
        tests = json.load(f)
    ranks, totals = [], {}
    by_type = {}
    for q in tests:
        correct = set().union(*[find(a["title"], a["year"]) for a in q["answers"]])
        if not correct:
            continue
        results, times = engine.search(q["query"])
        rank = next((r + 1 for r, x in enumerate(results) if x["index"] in correct), None)
        ranks.append(rank)
        by_type.setdefault(q["type"], []).append(rank)
        for k_, v in times.items():
            totals.setdefault(k_, []).append(v)

    def line(name, rs):
        n = len(rs)
        print(f"{name:<9}{n:>4}{sum(r is not None for r in rs) / n:>11.2f}{sum(r == 1 for r in rs) / n:>8.2f}"
              f"{sum(1 / r for r in rs if r) / n:>8.2f}")

    print(f"\nReranker: {RERANKER}")
    print(f"{'type':<9}{'n':>4}{'top10':>11}{'1st':>8}{'MRR':>8}")
    for t in ["word", "meaning", "judge"]:
        if t in by_type:
            line(t, by_type[t])
    line("ALL", ranks)
    print("\nSpeed per question (middle):  " + "  ".join(
        f"{k_.replace('_ms', '')} {sorted(v)[len(v) // 2]:.0f} ms" for k_, v in totals.items()))


if __name__ == "__main__":
    engine = HybridSearch()
    if len(sys.argv) > 1 and sys.argv[1] == "--eval":
        evaluate(engine)
    else:
        question = " ".join(sys.argv[1:]) or "a man relives the same day again and again"
        results, times = engine.search(question)
        print(f"\nQuestion: {question}\n")
        for r, x in enumerate(results, 1):
            s = x["scores"]
            print(f"{r:>2}. {x['title']} ({x['year']})   bm25 #{s['bm25_place'] or '-'}  "
                  f"dense #{s['dense_place'] or '-'}  fusion #{s['fusion_place']}  rerank {s['rerank']}")
        print(f"\nTime: {times}")