"""
Reranker: the "careful judge". It reads the question TOGETHER with each shortlisted movie
and gives a new score. The shortlist comes from fusion (BM25 + dense, 3:1).

Input:  data/merged/movies.json
        data/index/dense_bge_small.npy   (made by search/dense.py)
        eval/test_set.json
Output: eval/results_rerank_<name>.json  (top 10 for every question, to read by hand)

Run from the main project folder:
    python search/rerank.py

It tries two rerankers and prints both next to BM25 and fusion:
  - ms-marco MiniLM : small and fast, trained on web-search questions
  - bge-reranker-base: bigger and slower, usually better
"""
import json
import sys
import time

import bm25s
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

sys.path.insert(0, "cleaning")
from common import make_loose_key, make_title_key  # noqa: E402

MOVIES = "data/merged/movies.json"
TEST_SET = "eval/test_set.json"
DENSE_INDEX = "data/index/dense_bge_small.npy"

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

RERANKERS = {
    "msmarco_minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge_reranker_base": "BAAI/bge-reranker-base",
}

CANDIDATES = 100          # each search hands in this many
FUSION_WEIGHTS = (3, 1)   # BM25 : dense  (the best mix from fusion.py)
RRF_K = 60
SHORTLIST = 30            # the reranker reads only the top 30 of the fused list
RERANK_WORDS = 300        # words of movie text the reranker reads (it can't read whole plots)
TOP_K = 10


# ---------- Step 1: load movies ----------
t0 = time.perf_counter()
with open(MOVIES, encoding="utf-8") as f:
    movies = json.load(f)
print(f"Loaded {len(movies)} movies in {time.perf_counter() - t0:.1f} s")

try:
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    device = "cpu"


# ---------- Step 2: BM25 + dense + fusion (same as fusion.py) ----------
def movie_text(m):
    parts = [m["title"], *m["other_titles"], *m["genres"], *m["keywords"],
             m["tagline"], m["tmdb_overview"], m["cmu_plot"], m["wiki_plot"]]
    return " ".join(p for p in parts if p)

bm25 = bm25s.BM25()
bm25.index(bm25s.tokenize([movie_text(m) for m in movies], stopwords="en", show_progress=False),
           show_progress=False)
emb = np.load(DENSE_INDEX).astype(np.float32)
dense_model = SentenceTransformer(DENSE_MODEL, device=device)
print(f"BM25 + dense ready (device: {device})")

def bm25_list(query):
    docs, _ = bm25.retrieve(bm25s.tokenize([query], stopwords="en", show_progress=False),
                            k=CANDIDATES, show_progress=False)
    return [int(d) for d in docs[0]]

def dense_list(query):
    q = dense_model.encode([QUERY_PREFIX + query], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = emb @ q
    top = np.argpartition(-scores, CANDIDATES)[:CANDIDATES]
    return [int(d) for d in top[np.argsort(-scores[top])]]

def fused_list(query):
    points = {}
    for ranked, weight in [(bm25_list(query), FUSION_WEIGHTS[0]), (dense_list(query), FUSION_WEIGHTS[1])]:
        for place, d in enumerate(ranked, start=1):
            points[d] = points.get(d, 0) + weight / (RRF_K + place)
    return sorted(points, key=points.get, reverse=True)


# ---------- Step 3: the text the reranker reads for each movie ----------
# Title + year + overview + the start of the plot, cut to RERANK_WORDS words.
def judge_text(m):
    story = " ".join(p for p in [m["tmdb_overview"], m["wiki_plot"] or m["cmu_plot"]] if p)
    text = f"{m['title']} ({m['year']}). {story}"
    return " ".join(text.split()[:RERANK_WORDS])


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


def rank_of(ranked, correct):
    return next((r + 1 for r, x in enumerate(ranked[:TOP_K]) if x in correct), None)

def score(ranks):
    n = len(ranks)
    return (sum(r == 1 for r in ranks) / n, sum(r is not None for r in ranks) / n,
            sum(1 / r for r in ranks if r) / n)


# ---------- Step 5: shortlists from BM25 and fusion (once) ----------
bm25_ranks, fusion_ranks, shortlists = {}, {}, {}
in_shortlist = 0
for q in tests:
    shortlists[q["qid"]] = fused_list(q["query"])[:SHORTLIST]
    bm25_ranks[q["qid"]] = rank_of(bm25_list(q["query"]), q["correct"])
    fusion_ranks[q["qid"]] = rank_of(shortlists[q["qid"]], q["correct"])
    in_shortlist += any(x in q["correct"] for x in shortlists[q["qid"]])
print(f"Right movie is in the fusion top {SHORTLIST} for {in_shortlist}/{len(tests)} questions "
      f"(the reranker can only fix these)")


# ---------- Step 6: rerank the shortlist with each reranker ----------
all_ranks = {"BM25": bm25_ranks, "Fusion 3:1": fusion_ranks}
speeds = {}
for name, model_name in RERANKERS.items():
    judge = CrossEncoder(model_name, device=device, max_length=512)
    ranks, times, results = {}, [], []
    for q in tests:
        short = shortlists[q["qid"]]
        t0 = time.perf_counter()
        scores = judge.predict([(q["query"], judge_text(movies[d])) for d in short],
                               batch_size=SHORTLIST, show_progress_bar=False)
        times.append((time.perf_counter() - t0) * 1000)
        order = [short[i] for i in np.argsort(-np.asarray(scores))]
        ranks[q["qid"]] = rank_of(order, q["correct"])
        results.append({
            "qid": q["qid"], "type": q["type"], "query": q["query"],
            "answer": q["answers"][0]["title"],
            "rank_before": fusion_ranks[q["qid"]], "rank_after": ranks[q["qid"]],
            "top10": [f"{movies[d]['title']} ({movies[d]['year']})  fusion_place={short.index(d) + 1}"
                      for d in order[:TOP_K]],
        })
    all_ranks[f"Rerank {name}"] = ranks
    speeds[name] = sorted(times)[len(times) // 2]
    with open(f"eval/results_rerank_{name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Reranked with {model_name}: middle {speeds[name]:.0f} ms per question")


# ---------- Step 7: the numbers ----------
print("\nRight movie in the top 10 | in 1st place:")
print(f"{'':<26}" + "".join(f"{t:>14}" for t in ["word", "meaning", "judge", "ALL"]) + f"{'MRR':>7}")
for name, ranks in all_ranks.items():
    line = f"{name:<26}"
    for t in ["word", "meaning", "judge", "ALL"]:
        h1, h10, _ = score([ranks[q["qid"]] for q in tests if t == "ALL" or q["type"] == t])
        line += f"{f'{h10:.2f} | {h1:.2f}':>14}"
    print(line + f"{score(list(ranks.values()))[2]:>7.2f}")

best = max(RERANKERS, key=lambda n: score(list(all_ranks[f'Rerank {n}'].values()))[2])
print(f"\nBest reranker: {best}")
print(f"\nWhat the best reranker changed (fusion place -> after reranking; - = not in top 10):")
for q in tests:
    a, b = fusion_ranks[q["qid"]], all_ranks[f"Rerank {best}"][q["qid"]]
    if a != b:
        tag = "BETTER" if (b or 99) < (a or 99) else "worse "
        print(f"   {tag}  {q['qid']} [{q['type']:<7}] {a or '-':>2} -> {b or '-':>2}   {q['answers'][0]['title']}")
print("\nSaved: eval/results_rerank_<name>.json")