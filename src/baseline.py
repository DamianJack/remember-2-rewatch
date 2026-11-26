"""Step 2 - Baselines: the simplest ways to find a movie from a vague description.

  tfidf   TF-IDF vectors + cosine similarity between the description and every plot
  bm25    BM25: like TF-IDF, but repeated words count less and long plots aren't favoured
  prompt  plain prompt: ask the local LLM (Qwen via Ollama) "which movie is this?";
          it answers from memory and never sees our data

Every method has rank(query) -> row positions of `movies`, best match first.
BM25 is also the keyword half of the hybrid search (hybrid_search.py).

Try:  python src/baseline.py "a shark keeps attacking swimmers at a beach town"
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

from prepare_data import norm_title

ROOT = Path(__file__).resolve().parent.parent


def load_movies() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "movies.parquet")


def tokenize(text: str) -> list[str]:
    """lowercase words, without stop words ('the', 'and', ...) and single letters"""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in ENGLISH_STOP_WORDS]


class TfidfSearch:
    def __init__(self, movies: pd.DataFrame):
        self.vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                   sublinear_tf=True, min_df=2)
        self.matrix = self.vec.fit_transform(movies["plot"])

    def rank(self, query: str) -> np.ndarray:
        scores = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        return np.argsort(-scores, kind="stable")


class BM25Search:
    """Okapi BM25. The weight of every (plot, word) pair is computed once into a sparse matrix,
    so a query is just a sum of a few columns (milliseconds for 120k plots).
      weight = idf(word) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * plot_length / average_length))"""

    def __init__(self, movies: pd.DataFrame, k1: float = 1.5, b: float = 0.75):
        self.vec = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        tf = self.vec.fit_transform(movies["plot"]).astype(np.float32)  # plots x words, word counts
        length = np.asarray(tf.sum(axis=1)).ravel()
        df = np.bincount(tf.indices, minlength=tf.shape[1])  # in how many plots each word occurs
        idf = np.log(1 + (len(length) - df + 0.5) / (df + 0.5))
        row = np.repeat(np.arange(len(length)), np.diff(tf.indptr))
        tf.data = idf[tf.indices] * tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * length[row] / length.mean()))
        self.weights = tf.tocsc()

    def scores(self, query: str) -> np.ndarray:
        cols = [self.vec.vocabulary_[w] for w in tokenize(query) if w in self.vec.vocabulary_]
        return np.asarray(self.weights[:, cols].sum(axis=1)).ravel()

    def rank(self, query: str) -> np.ndarray:
        return np.argsort(-self.scores(query), kind="stable")


# ---------------------------------------------------------------- plain prompt (local LLM)
MODEL = "qwen2.5:7b"
CACHE = ROOT / "eval" / "llm_cache.json"
_cache: dict | None = None


def ask_llm(system: str, user: str, schema: dict) -> dict:
    """One question to the local model; `schema` forces the reply to be JSON of that shape.
    Replies are cached in eval/llm_cache.json, so a re-run costs nothing."""
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    key = hashlib.md5(json.dumps([MODEL, system, user, schema]).encode()).hexdigest()
    if key not in _cache:
        try:
            r = requests.post("http://localhost:11434/api/chat", timeout=180, json={
                "model": MODEL, "stream": False, "format": schema,
                "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 400, "repeat_penalty": 1.1},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        except requests.ConnectionError:
            raise SystemExit("Ollama isn't running. Start it with: ollama serve")
        if r.status_code == 404:
            raise SystemExit(f"Model not downloaded. Run: ollama pull {MODEL}")
        r.raise_for_status()
        _cache[key] = json.loads(r.json()["message"]["content"])
        CACHE.write_text(json.dumps(_cache))
    return _cache[key]


PROMPT_SYSTEM = ("You are a film expert. Someone vaguely remembers a movie and describes it. "
                 "Give your 5 best guesses for which movie it is, most likely first, "
                 "with the original release year (0 if you don't know it).")
PROMPT_SCHEMA = {"type": "object", "required": ["guesses"], "properties": {"guesses": {
    "type": "array", "minItems": 1, "maxItems": 5, "items": {
        "type": "object", "required": ["title", "year"],
        "properties": {"title": {"type": "string"}, "year": {"type": "integer"}}}}}}


class PromptSearch:
    """Only the movies the model names (max 5) are ranked; anything else counts as not found."""

    def __init__(self, movies: pd.DataFrame):
        self.titles = movies["title"].map(norm_title).to_numpy()
        self.years = movies["year"].fillna(0).to_numpy()

    def rank(self, query: str) -> np.ndarray:
        guesses = ask_llm(PROMPT_SYSTEM, f"Description: {query}\n\nWhich movie is this?", PROMPT_SCHEMA)["guesses"]
        found = []
        for g in guesses:  # map each guess to our movie with that title and year (+-1)
            rows = np.flatnonzero(self.titles == norm_title(g["title"]))
            close = rows[abs(self.years[rows] - g["year"]) <= 1]
            found += [r for r in (close if len(close) else rows[:1]) if r not in found]
        return np.array(found, dtype=int)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) or "a shark keeps attacking swimmers at a beach town"
    movies = load_movies()
    for name, method in [("tfidf", TfidfSearch), ("bm25", BM25Search), ("prompt", PromptSearch)]:
        top = method(movies).rank(query)[:5]
        print(f"{name:7s}", " | ".join(f"{t} ({y})" for t, y in movies.iloc[top][["title", "year"]].values))
