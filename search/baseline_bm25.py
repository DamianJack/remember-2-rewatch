"""
Baseline: plain BM25 word search on the merged movies, measured on the hand-checked test set.

Input:  data/merged/movies.json   (119,025 movies)
        eval/test_set.json         (50 questions + correct movies)
Output: eval/results_bm25.json     (top 10 for every question, to read by hand)

Run from the main project folder:
    python search/bm25_baseline.py
"""
import json
import sys
import time

import bm25s

sys.path.insert(0, "cleaning")
from common import make_loose_key, make_title_key  # noqa: E402

MOVIES = "data/merged/movies.json"
TEST_SET = "eval/test_set.json"
OUT = "eval/results_bm25.json"
TOP_K = 10


# ---------- Step 1: load the movies ----------
t0 = time.perf_counter()
with open(MOVIES, encoding="utf-8") as f:
    movies = json.load(f)
print(f"Loaded {len(movies)} movies in {time.perf_counter() - t0:.1f} s")


# ---------- Step 2: one text per movie (everything we have, glued together) ----------
def movie_text(m):
    parts = [m["title"], *m["other_titles"], *m["genres"], *m["keywords"],
             m["tagline"], m["tmdb_overview"], m["cmu_plot"], m["wiki_plot"]]
    return " ".join(p for p in parts if p)

corpus = [movie_text(m) for m in movies]


# ---------- Step 3: build the BM25 index ----------
t0 = time.perf_counter()
tokens = bm25s.tokenize(corpus, stopwords="en", show_progress=False)
retriever = bm25s.BM25()
retriever.index(tokens, show_progress=False)
print(f"Built BM25 index in {time.perf_counter() - t0:.1f} s")


# ---------- Step 4: find the correct movie(s) for every test question ----------
# The test set names the answer by title + year. We look it up in the data:
# same title (normal or loose spelling, main or other title), year +-1.
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

not_found = []
for q in tests:
    q["correct"] = set()
    for a in q["answers"]:
        q["correct"] |= find_movie(a["title"], a["year"])
    if not q["correct"]:
        not_found.append(q)

print(f"\nTest questions: {len(tests)}   answer not found in the data: {len(not_found)}")
for q in not_found:
    print(f"   {q['qid']}: {q['answers'][0]['title']} ({q['answers'][0]['year']})")
tests = [q for q in tests if q["correct"]]


# ---------- Step 5: ask every question, see where the right movie lands ----------
results = []
for q in tests:
    t0 = time.perf_counter()
    q_tokens = bm25s.tokenize([q["query"]], stopwords="en", show_progress=False)
    docs, scores = retriever.retrieve(q_tokens, k=TOP_K, show_progress=False)
    ms = (time.perf_counter() - t0) * 1000

    top = [int(d) for d in docs[0]]
    rank = next((r + 1 for r, d in enumerate(top) if d in q["correct"]), None)
    results.append({
        "qid": q["qid"], "type": q["type"], "query": q["query"],
        "answer": q["answers"][0]["title"],
        "rank": rank, "latency_ms": round(ms, 1),
        "top10": [f"{movies[d]['title']} ({movies[d]['year']})  score={s:.1f}"
                  for d, s in zip(top, scores[0])],
    })


# ---------- Step 6: the numbers ----------
def summary(rows):
    n = len(rows)
    hit1 = sum(r["rank"] == 1 for r in rows) / n
    hit10 = sum(r["rank"] is not None for r in rows) / n
    mrr = sum(1 / r["rank"] for r in rows if r["rank"]) / n
    return n, hit1, hit10, mrr

print(f"\n{'type':<9}{'n':>4}{'Hit@1':>9}{'Recall@10':>11}{'MRR@10':>9}")
for t in ["word", "meaning", "judge", "ALL"]:
    rows = results if t == "ALL" else [r for r in results if r["type"] == t]
    if rows:
        n, h1, h10, mrr = summary(rows)
        print(f"{t:<9}{n:>4}{h1:>9.2f}{h10:>11.2f}{mrr:>9.2f}")

lat = sorted(r["latency_ms"] for r in results)
print(f"\nSpeed per question: middle {lat[len(lat) // 2]:.1f} ms, slowest {lat[-1]:.1f} ms")

print("\nMissed (right movie not in the top 10):")
for r in results:
    if r["rank"] is None:
        print(f"   {r['qid']} [{r['type']}] {r['answer']}  <- got: {r['top10'][0]}")


# ---------- Step 7: save everything, to read the outputs by hand (rule 4) ----------
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nSaved to", OUT)