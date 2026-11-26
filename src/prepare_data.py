"""Step 1 - Organise the data: every source merged into ONE table, one row per movie.

Sources (all in data/):
  CMU Movie Summary Corpus   movie.metadata.tsv, plot_summaries.txt, character.metadata.tsv
  Kaggle Wikipedia plots     wiki_movie_plots_deduped.csv
  TMDB                       TMDB_movie_dataset_v11.csv
The same movie in two sources is matched by normalised title (or TMDB's original title)
+ release year (+-1).

One row per movie, one column per field:
  movie_id   stable id: cmu_<wikipedia id>, kg_<title>_<year> or tmdb_<id>
  title, year
  genres     list, from ONE fixed vocabulary (GENRES), so a genre in a query matches exactly
  countries  list
  people     list of actors and directors
  keywords   list of TMDB tags ("shark", "time travel", ...)
  votes      TMDB vote count (how well known the movie is)
  plot       the longest plot we have (Wikipedia > CMU > TMDB overview), cleaned:
             no citation marks like [1], no HTML, single spaces
Only movies with a plot are kept (nothing else can be searched), and duplicates are removed:
the same plot text twice, or the same title + year twice (the best-known copy is kept).

Saved as data/movies.parquet (zstd-compressed): keeps the list columns, loads in 0.2 s.
Run:  python src/prepare_data.py
"""

from __future__ import annotations

import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

# fixed genre vocabulary (TMDB's genres) and the patterns that map each source's raw labels onto it
GENRES = {
    "action": r"action|martial arts", "adventure": r"adventure|swashbuckler",
    "animation": r"animat|anime|cartoon", "comedy": r"comed|parody|satire|spoof",
    "crime": r"crime|gangster|heist|detective|noir|mafia", "documentary": r"documentar",
    "drama": r"drama", "family": r"family|children", "fantasy": r"fantasy|fairy tale",
    "history": r"histor|biograph|biopic|period piece", "horror": r"horror|slasher|zombie",
    "music": r"music", "mystery": r"myster|whodunit", "romance": r"roman",
    "science fiction": r"sci-?fi|science", "thriller": r"thriller|suspense",
    "war": r"\bwar\b", "western": r"western",
}


def norm_title(t) -> str:
    """'The Matrix' -> 'matrix', so titles from different sources compare equal."""
    t = unicodedata.normalize("NFKD", str(t)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "", re.sub(r"^(the|a|an)\s+", "", t))


def clean_text(t):
    """'The film [1] is &amp; was <i>great</i>\n\n' -> 'The film is & was great'"""
    if not isinstance(t, str):
        return t
    t = re.sub(r"<ref[^}>]*(\}\}|>)|\{\{[^}]*\}\}", " ", t)      # Wikipedia markup: <ref ...}}, {{...}}
    t = re.sub(r"<[^>]+>", " ", html.unescape(t))                # HTML tags and entities
    t = re.sub(r"\[(\d+|citation needed|note \d+)\]", "", t)     # Wikipedia citation marks
    return re.sub(r"\s+", " ", t).strip() or None                # newlines, double spaces


def to_genres(raw: list[str]) -> list[str]:
    text = " | ".join(raw).lower()
    return [g for g, pattern in GENRES.items() if re.search(pattern, text)]


def as_list(v) -> list:
    return list(v) if isinstance(v, list) else []


def split_names(s: str) -> list[str]:
    return [n.strip() for n in re.split(r",|\band\b", s) if n.strip() and n.strip() != "Unknown"]


# ---------------------------------------------------------------- the three sources
def load_cmu() -> pd.DataFrame:
    cols = ["wiki_id", "fb_id", "title", "date", "box_office", "runtime", "languages", "countries", "genres"]
    m = pd.read_csv(DATA / "movie.metadata.tsv", sep="\t", names=cols)
    freebase = lambda s: list(json.loads(s).values()) if isinstance(s, str) else []  # '{"/m/x": "Drama"}'
    with open(DATA / "plot_summaries.txt", encoding="utf-8") as f:
        plots = dict(line.rstrip("\n").partition("\t")[::2] for line in f)
    chars = pd.read_csv(DATA / "character.metadata.tsv", sep="\t", header=None,
                        usecols=[0, 8], names=["wiki_id", "actor"])
    actors = chars.dropna().groupby("wiki_id")["actor"].agg(lambda s: list(dict.fromkeys(s)))
    return pd.DataFrame({
        "movie_id": "cmu_" + m["wiki_id"].astype(str), "title": m["title"],
        "year": pd.to_numeric(m["date"].str[:4], errors="coerce"),
        "cmu_plot": m["wiki_id"].astype(str).map(plots),
        "cmu_genres": m["genres"].map(freebase), "cmu_countries": m["countries"].map(freebase),
        "cmu_people": m["wiki_id"].map(actors)})


def load_wiki() -> pd.DataFrame:
    w = pd.read_csv(DATA / "wiki_movie_plots_deduped.csv")
    w.columns = ["year", "title", "origin", "director", "cast", "genre", "url", "plot"]
    return pd.DataFrame({
        "movie_id": "kg_" + w["title"].map(norm_title) + "_" + w["year"].astype(str),
        "title": w["title"], "year": w["year"].astype(float), "wiki_plot": w["plot"],
        "wiki_genres": w["genre"].map(lambda g: [] if g == "unknown" else [g]),
        "wiki_people": (w["cast"].fillna("") + "," + w["director"].fillna("")).map(split_names),
    }).drop_duplicates("movie_id")


def load_tmdb() -> pd.DataFrame:
    cols = ["id", "title", "original_title", "release_date", "status", "adult", "overview", "genres",
            "production_countries", "keywords", "vote_count"]
    t = pd.read_csv(DATA / "TMDB_movie_dataset_v11.csv", usecols=cols, low_memory=False)
    t = t[(t["status"] == "Released") & ~t["adult"]]
    commas = lambda s: [x.strip() for x in s.split(",")] if isinstance(s, str) else []
    return pd.DataFrame({
        "movie_id": "tmdb_" + t["id"].astype(str), "title": t["title"], "alt_title": t["original_title"],
        "year": pd.to_datetime(t["release_date"], errors="coerce").dt.year, "tmdb_plot": t["overview"],
        "tmdb_genres": t["genres"].map(commas), "tmdb_countries": t["production_countries"].map(commas),
        "keywords": t["keywords"].map(commas), "votes": t["vote_count"]})


# ---------------------------------------------------------------- merge
def merge(base: pd.DataFrame, new: pd.DataFrame, add_unmatched=None) -> pd.DataFrame:
    """Attach each row of `new` to the `base` movie with the same normalised title (or its
    `alt_title`, e.g. TMDB's original title) and a year within 1 (sources often disagree by a
    year). Unmatched rows become new movies (only where `add_unmatched` is True, if given)."""
    base = base.reset_index(drop=True)
    titles = [new["title"]] + ([new["alt_title"]] if "alt_title" in new else [])
    left = pd.concat([new.assign(k=t.map(norm_title)) for t in titles]).dropna(subset=["year"]).sort_values("year")
    right = base.assign(k=base["title"].map(norm_title), row=base.index).dropna(subset=["year"])
    m = pd.merge_asof(left, right[["k", "year", "row"]].sort_values("year"),
                      on="year", by="k", tolerance=1, direction="nearest")
    if "votes" in m:  # several candidates for one movie: keep the best-known
        m = m.sort_values("votes", ascending=False)
    matched = m.dropna(subset=["row"]).drop_duplicates("row").drop_duplicates("movie_id")
    data_cols = [c for c in new.columns if c not in ("movie_id", "title", "alt_title", "year")]
    base = base.join(matched.set_index(matched["row"].astype(int))[data_cols])
    rest = new[~new["movie_id"].isin(matched["movie_id"])]
    if add_unmatched is not None:
        rest = rest[add_unmatched.loc[rest.index]]
    return pd.concat([base, rest.drop(columns="alt_title", errors="ignore")], ignore_index=True)


def build() -> pd.DataFrame:
    tmdb = load_tmdb()
    # TMDB has 1M+ stubs: only add TMDB-only movies that are somewhat known and have a real overview
    known = (tmdb["votes"] >= 10) & (tmdb["tmdb_plot"].fillna("").str.split().str.len() >= 15)
    df = merge(merge(load_cmu(), load_wiki()), tmdb, add_unmatched=known)

    lists = {c: df[c].map(as_list) for c in ["cmu_genres", "wiki_genres", "tmdb_genres", "cmu_countries",
                                             "tmdb_countries", "cmu_people", "wiki_people", "keywords"]}
    movies = pd.DataFrame({
        "movie_id": df["movie_id"], "title": df["title"], "year": df["year"].astype("Int64"),
        "genres": [to_genres(a + b + c) for a, b, c in
                   zip(lists["cmu_genres"], lists["wiki_genres"], lists["tmdb_genres"])],
        "countries": [sorted(set(a + b)) for a, b in zip(lists["cmu_countries"], lists["tmdb_countries"])],
        "people": [list(dict.fromkeys(a + b)) for a, b in zip(lists["cmu_people"], lists["wiki_people"])],
        "keywords": lists["keywords"], "votes": df["votes"].fillna(0).astype(int),
        "plot": df["wiki_plot"].map(clean_text).fillna(df["cmu_plot"].map(clean_text))
                               .fillna(df["tmdb_plot"].map(clean_text)),
    })
    movies = movies[movies["plot"].notna()]

    # duplicates: same plot text, or same title + year -> keep the best-known, then longest plot
    key = movies["title"].map(norm_title) + "_" + movies["year"].astype(str)
    best_first = movies.assign(n=movies["plot"].str.len()).sort_values(["votes", "n"], ascending=False)
    keep = best_first.drop_duplicates("plot")
    keep = keep[~(key[keep.index].duplicated() & keep["year"].notna())]
    print(f"removed {len(movies) - len(keep):,} duplicates")
    return movies.loc[sorted(keep.index)].reset_index(drop=True)


if __name__ == "__main__":
    movies = build()
    movies.to_parquet(DATA / "movies.parquet", index=False, compression="zstd")
    print(f"saved data/movies.parquet: {len(movies):,} movies")
    print(movies.iloc[0])
