"""
Dense search (meaning search), measured on the same test set as BM25.

Input:  data/merged/movies.json
        eval/test_set.json
Output: data/index/dense_<model>.npy    (the "fingerprints", made once per model, then reused)
        eval/results_dense_<model>.json  (top 10 for every question, to read by hand)

To try another model, only change the MODEL SETTINGS block below.

Run from the main project folder:
    python search/dense_search.py

The FIRST run makes the fingerprints for all movies (takes a while: roughly 15-40 min on a Mac).
Every run after that just loads them (a few seconds).
"""
import json
import os
import sys
import time

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "cleaning")
from common import make_loose_key, make_title_key  # noqa: E402

MOVIES = "data/merged/movies.json"
TEST_SET = "eval/test_set.json"
INDEX_DIR = "data/index"
BM25_RESULTS = "eval/results_bm25.json"        # to compare with the baseline, if it exists
OLD_DENSE_RESULTS = "eval/results_dense.json"  # the first MiniLM run, to compare with

# ---------- MODEL SETTINGS ----------
# bge-small: made for "short question -> longer text" search.
# It needs this instruction in front of every QUESTION (not in front of the movie texts).
# The user never types it; the code adds it automatically.
MODEL = "BAAI/bge-small-en-v1.5"
SHORT_NAME = "bge_small"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# (the first run used these:)
# MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# SHORT_NAME = "minilm"
# QUERY_PREFIX = ""

INDEX = f"{INDEX_DIR}/dense_{SHORT_NAME}.npy"
OUT = f"eval/results_dense_{SHORT_NAME}.json"
TOP_K = 10
PLOT_WORDS = 200       # if a movie has no overview, use this many words of its plot


# ---------- Step 1: load the movies ----------
t0 = time.perf_counter()
with open(MOVIES, encoding="utf-8") as f:
    movies = json.load(f)
print(f"Loaded {len(movies)} movies in {time.perf_counter() - t0:.1f} s")


# ---------- Step 2: one SHORT text per movie (the model reads only ~200 words) ----------
# Title first, then the overview (or the start of the plot), then genres and keywords.
# If the text is too long, the model cuts off the END, so the most useful part goes first.
def short_text(m):
    story = m["tmdb_overview"] or " ".join((m["wiki_plot"] or m["cmu_plot"]).split()[:PLOT_WORDS])
    parts = [m["title"], story, ", ".join(m["genres"]), ", ".join(m["keywords"])]
    return ". ".join(p for p in parts if p)

texts = [short_text(m) for m in movies]


# ---------- Step 3: the model ----------
# On a Mac with Apple chip, "mps" uses the graphics chip and is much faster than the CPU.
try:
    import torch
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    device = "cpu"
model = SentenceTransformer(MODEL, device=device)
print(f"Model: {MODEL} on {device}")


# ---------- Step 4: make the fingerprints (only the first time), or load them ----------
IDS_FILE = INDEX.replace(".npy", "_ids.json")   # which movie each fingerprint belongs to
movie_ids = [m["id"] for m in movies]
saved_ids = None
if os.path.exists(IDS_FILE):
    with open(IDS_FILE, encoding="utf-8") as f:
        saved_ids = json.load(f)

if os.path.exists(INDEX) and saved_ids == movie_ids:
    emb = np.load(INDEX).astype(np.float32)
    print(f"Loaded fingerprints from {INDEX}")
else:
    if os.path.exists(INDEX):
        print("movies.json has changed since the fingerprints were made -> rebuilding them.")
    emb = None

if emb is None:
    print("Making fingerprints for all movies (only when movies.json is new or changed)...")
    t0 = time.perf_counter()
    emb = model.encode(texts, batch_size=128, show_progress_bar=True,
                       normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(INDEX, emb.astype(np.float16))     # float16 = half the file size, same results
    with open(IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(movie_ids, f)
    print(f"Made and saved {emb.shape[0]} fingerprints of {emb.shape[1]} numbers "
          f"in {(time.perf_counter() - t0) / 60:.1f} min")


# ---------- Step 5: find the correct movie(s) for every test question (same as BM25) ----------
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


# ---------- Step 6: ask every question ----------
# Question -> fingerprint -> compare with all movie fingerprints -> 10 closest.
results = []
for q in tests:
    t0 = time.perf_counter()
    q_emb = model.encode([QUERY_PREFIX + q["query"]], normalize_embeddings=True, convert_to_numpy=True)[0]
    scores = emb @ q_emb                                   # closeness to every movie (1 = same meaning)
    top = np.argpartition(-scores, TOP_K)[:TOP_K]          # the 10 closest (not yet in order)
    top = top[np.argsort(-scores[top])]                    # put them in order
    ms = (time.perf_counter() - t0) * 1000

    top = [int(d) for d in top]
    rank = next((r + 1 for r, d in enumerate(top) if d in q["correct"]), None)
    results.append({
        "qid": q["qid"], "type": q["type"], "query": q["query"],
        "answer": q["answers"][0]["title"],
        "rank": rank, "latency_ms": round(ms, 1),
        "top10": [f"{movies[d]['title']} ({movies[d]['year']})  score={scores[d]:.3f}" for d in top],
    })


# ---------- Step 7: the numbers, next to the BM25 baseline ----------
def summary(rows):
    n = len(rows)
    return (n, sum(r["rank"] == 1 for r in rows) / n,
            sum(r["rank"] is not None for r in rows) / n,
            sum(1 / r["rank"] for r in rows if r["rank"]) / n)

def load_results(path):
    if os.path.exists(path) and path != OUT:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

bm25 = load_results(BM25_RESULTS)
others = [(f"DENSE {SHORT_NAME}", results), ("MiniLM (1st run)", load_results(OLD_DENSE_RESULTS)), ("BM25", bm25)]
others = [(name, rows) for name, rows in others if rows]

print("\nRight movie in the top 10 (Recall@10) and in 1st place (Hit@1):")
print(f"{'type':<9}{'n':>4}" + "".join(f"{name:>24}" for name, _ in others))
for t in ["word", "meaning", "judge", "ALL"]:
    line = f"{t:<9}"
    for k, (name, rows_all) in enumerate(others):
        rows = rows_all if t == "ALL" else [r for r in rows_all if r["type"] == t]
        n, h1, h10, mrr = summary(rows)
        if k == 0:
            line += f"{n:>4}"
        line += f"{'top10 ' + format(h10, '.2f') + ' | 1st ' + format(h1, '.2f'):>24}"
    print(line)

lat = sorted(r["latency_ms"] for r in results)
print(f"\nSpeed per question: middle {lat[len(lat) // 2]:.1f} ms, slowest {lat[-1]:.1f} ms")

print("\nMissed by dense (right movie not in the top 10):")
for r in results:
    if r["rank"] is None:
        print(f"   {r['qid']} [{r['type']}] {r['answer']}  <- got: {r['top10'][0]}")

if bm25:
    b_rank = {r["qid"]: r["rank"] for r in bm25}
    print("\nWhere they disagree (rank of the right movie; - = not in top 10):")
    for r in results:
        d, b = r["rank"], b_rank.get(r["qid"])
        if (d is None) != (b is None) or (d and b and abs(d - b) >= 3):
            winner = "DENSE" if (d or 99) < (b or 99) else "BM25"
            print(f"   {r['qid']} [{r['type']:<7}] dense {d or '-':>2} | bm25 {b or '-':>2}  -> {winner} wins   {r['answer']}")


# ---------- Step 8: save, to read the outputs by hand ----------
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nSaved to", OUT)