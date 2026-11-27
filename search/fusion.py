"""
Fusion: mix the BM25 list and the dense list into ONE list (Reciprocal Rank Fusion, RRF).

Input:  data/merged/movies.json
        data/index/dense_bge_small.npy   (made by search/dense.py)
        eval/test_set.json
Output: eval/results_fusion.json         (top 10 for every question, best weight setting)

Run from the main project folder:
    python search/fusion.py

How RRF works:
  Each search gives its own ranked list. A movie gets points for its PLACE in each list:
      points = weight / (60 + place)
  1st place gives the most points, 100th place very few. We add up the points from both lists.
  A movie that is high in BOTH lists wins. Raw scores (BM25 "12.4", dense "0.61") are ignored,
  because they are on different scales.
"""
import json
import sys
import time

import bm25s
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "cleaning")
from common import make_loose_key, make_title_key  # noqa: E402

MOVIES = "data/merged/movies.json"
TEST_SET = "eval/test_set.json"
DENSE_INDEX = "data/index/dense_bge_small.npy"
OUT = "eval/results_fusion.json"

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

CANDIDATES = 100     # how many movies each search hands in
RRF_K = 60           # the standard RRF constant (makes the points fall off gently)
TOP_K = 10

# Weight settings to test: (BM25 weight, dense weight)
WEIGHTS = [(1, 0), (0, 1), (1, 1), (2, 1), (3, 1), (5, 1)]


# ---------- Step 1: load movies ----------
t0 = time.perf_counter()
with open(MOVIES, encoding="utf-8") as f:
    movies = json.load(f)
print(f"Loaded {len(movies)} movies in {time.perf_counter() - t0:.1f} s")


# ---------- Step 2: BM25 (same as the baseline) ----------
def movie_text(m):
    parts = [m["title"], *m["other_titles"], *m["genres"], *m["keywords"],
             m["tagline"], m["tmdb_overview"], m["cmu_plot"], m["wiki_plot"]]
    return " ".join(p for p in parts if p)

t0 = time.perf_counter()
bm25 = bm25s.BM25()
bm25.index(bm25s.tokenize([movie_text(m) for m in movies], stopwords="en", show_progress=False),
           show_progress=False)
print(f"Built BM25 index in {time.perf_counter() - t0:.1f} s")


# ---------- Step 3: dense (load the saved fingerprints) ----------
emb = np.load(DENSE_INDEX).astype(np.float32)
try:
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    device = "cpu"
model = SentenceTransformer(MODEL, device=device)
print(f"Loaded {len(emb)} dense fingerprints, model on {device}")


def bm25_list(query):
    docs, _ = bm25.retrieve(bm25s.tokenize([query], stopwords="en", show_progress=False),
                            k=CANDIDATES, show_progress=False)
    return [int(d) for d in docs[0]]

def dense_list(query):
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = emb @ q
    top = np.argpartition(-scores, CANDIDATES)[:CANDIDATES]
    return [int(d) for d in top[np.argsort(-scores[top])]]

def rrf(lists_with_weights):
    """lists_with_weights: [(ranked list of movie numbers, weight), ...] -> one ranked list"""
    points = {}
    for ranked, weight in lists_with_weights:
        if weight == 0:
            continue
        for place, d in enumerate(ranked, start=1):
            points[d] = points.get(d, 0) + weight / (RRF_K + place)
    return sorted(points, key=points.get, reverse=True)


# ---------- Step 4: the correct movies for each question (same as before) ----------
by_key = {}
for i, m in enumerate(movies):
    years = {m["year"], *m["other_years"]}
    for title in [m["title"], *m["other_titles"]]:
        for key in {make_title_key(title), make_loose_key(make_title_key(title))}:
            for y in years:
                by_key.setdefault((key, y), set()).add(i)

def find_movie(title, year):
    key = make_title_key(title)
    found = set()
    for k in (key, make_loose_key(key)):
        for y in (year, year - 1, year + 1):
            found |= by_key.get((k, y), set())
    return found

with open(TEST_SET, encoding="utf-8") as f:
    tests = json.load(f)
for q in tests:
    q["correct"] = set()
    for a in q["answers"]:
        q["correct"] |= find_movie(a["title"], a["year"])
tests = [q for q in tests if q["correct"]]


# ---------- Step 5: run both searches ONCE per question, then try every weight setting ----------
lists = {}
for q in tests:
    t0 = time.perf_counter()
    b = bm25_list(q["query"])
    t1 = time.perf_counter()
    d = dense_list(q["query"])
    t2 = time.perf_counter()
    lists[q["qid"]] = (b, d, (t1 - t0) * 1000, (t2 - t1) * 1000)

def rank_of(ranked, correct):
    return next((r + 1 for r, x in enumerate(ranked[:TOP_K]) if x in correct), None)

def score(rows):
    n = len(rows)
    return (sum(r == 1 for r in rows) / n,
            sum(r is not None for r in rows) / n,
            sum(1 / r for r in rows if r) / n)

print("\nRight movie in the top 10 (top10) and in 1st place (1st), per weight setting:")
print(f"{'BM25:dense':<12}" + "".join(f"{t:>20}" for t in ["word", "meaning", "judge", "ALL"]) + f"{'MRR':>7}")
all_ranks = {}
for wb, wd in WEIGHTS:
    ranks = {q["qid"]: rank_of(rrf([(lists[q["qid"]][0], wb), (lists[q["qid"]][1], wd)]), q["correct"])
             for q in tests}
    all_ranks[(wb, wd)] = ranks
    line = f"{f'{wb}:{wd}':<12}"
    for t in ["word", "meaning", "judge", "ALL"]:
        rows = [ranks[q["qid"]] for q in tests if t == "ALL" or q["type"] == t]
        h1, h10, _ = score(rows)
        line += f"{f'{h10:.2f} | {h1:.2f}':>20}"
    _, _, mrr = score(list(ranks.values()))
    label = "  <- BM25 only" if wd == 0 else ("  <- dense only" if wb == 0 else "")
    print(line + f"{mrr:>7.2f}" + label)

# Best setting = highest MRR (rewards having the right movie high up), among the real mixes
best = max([w for w in WEIGHTS if w[0] and w[1]], key=lambda w: score(list(all_ranks[w].values()))[2])
print(f"\nBest mix: BM25:dense = {best[0]}:{best[1]}")

bm = [v[2] for v in lists.values()]
dm = [v[3] for v in lists.values()]
print(f"Speed per question (middle): BM25 {sorted(bm)[len(bm)//2]:.1f} ms + dense {sorted(dm)[len(dm)//2]:.1f} ms "
      f"+ mixing <1 ms")


# ---------- Step 6: what changed compared with BM25 alone ----------
only_bm25 = all_ranks[(1, 0)]
mix = all_ranks[best]
print(f"\nQuestions where the mix ({best[0]}:{best[1]}) differs from BM25 alone (- = not in top 10):")
for q in tests:
    a, b = only_bm25[q["qid"]], mix[q["qid"]]
    if a != b:
        better = "BETTER" if (b or 99) < (a or 99) else "worse "
        print(f"   {better}  {q['qid']} [{q['type']:<7}] bm25 {a or '-':>2} -> mix {b or '-':>2}   {q['answers'][0]['title']}")


# ---------- Step 7: save the best mix, to read by hand ----------
results = []
for q in tests:
    b, d, _, _ = lists[q["qid"]]
    fused = rrf([(b, best[0]), (d, best[1])])[:TOP_K]
    results.append({
        "qid": q["qid"], "type": q["type"], "query": q["query"],
        "answer": q["answers"][0]["title"], "rank": mix[q["qid"]],
        "top10": [f"{movies[x]['title']} ({movies[x]['year']})  "
                  f"bm25_place={b.index(x) + 1 if x in b else '-'}  dense_place={d.index(x) + 1 if x in d else '-'}"
                  for x in fused],
    })
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nSaved to", OUT)