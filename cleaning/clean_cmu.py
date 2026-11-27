"""
Clean the CMU Movie Summary Corpus.

Input:  data/raw/MovieSummaries/movie.metadata.tsv   (title, date, languages, countries, genres)
        data/raw/MovieSummaries/plot_summaries.txt   (plots)
Output: data/clean/cmu_clean.json

Run from the main project folder:
    python cleaning/clean_cmu.py
"""
import csv
import json
import re

import pandas as pd

from common import (FIRST_FILM_YEAR, clean_text, count_words, drop,
                    make_title_key, save_json)

FOLDER = "data/raw/MovieSummaries"
OUT = "data/clean/cmu_clean.json"

# Our decisions, kept in one place so they're easy to change
MIN_YEAR = FIRST_FILM_YEAR   # keep all years (same as Wikipedia)
MIN_WORDS = 30               # same as Wikipedia


# ---------- Step 1: load the two files ----------
meta = pd.read_csv(
    f"{FOLDER}/movie.metadata.tsv", sep="\t", header=None,
    names=["wiki_id", "freebase_id", "title", "release_date", "box_office",
           "runtime", "languages", "countries", "genres"],
)
plots = pd.read_csv(
    f"{FOLDER}/plot_summaries.txt", sep="\t", header=None,
    names=["wiki_id", "plot"],
    quoting=csv.QUOTE_NONE,   # plots contain " characters; read them as normal text
)
print("Start:", len(meta), "movies with details,", len(plots), "plots")


# ---------- Join the two files by wiki_id ----------
# "inner" = keep only movies that are in BOTH files
df = meta.merge(plots, on="wiki_id", how="inner")
print("  both plot and details:", len(df))


# ---------- Step 3: keep only the columns we need ----------
# (box_office, runtime and freebase_id are not needed for search)
df = df[["wiki_id", "title", "release_date", "languages", "countries", "genres", "plot"]]


# ---------- Step 5: clean the text ----------
df["title"] = df["title"].apply(clean_text)
df["plot"] = df["plot"].apply(clean_text)   # also fixes "Orlock ." -> "Orlock."


# Languages / countries / genres look like {"/m/02h40lc": "English Language"}.
# We only want the readable names.
def names(text):
    try:
        return list(json.loads(text).values())
    except (TypeError, ValueError):
        return []

def clean_languages(text):
    result = []
    for name in names(text):
        name = re.sub(r"\s+language$", "", name, flags=re.IGNORECASE).strip()
        if name and name.lower() != "silent film":     # "Silent film" is not a language
            result.append(name)
    return sorted(set(result))

def clean_countries(text):
    return sorted(set(n.strip() for n in names(text) if n.strip()))

def clean_genres(text):
    result = []
    for name in names(text):
        for part in name.split("/"):                     # "action/adventure" -> action, adventure
            part = part.lower().strip()
            part = re.sub(r"\s+(film|movie)s?$", "", part)   # "romance film" -> "romance"
            if part:
                result.append(part)
    return sorted(set(result))

df["languages"] = df["languages"].apply(clean_languages)
df["countries"] = df["countries"].apply(clean_countries)
df["genres"] = df["genres"].apply(clean_genres)


# ---------- Step 6: one clean year column ----------
# Dates are "1999" or "1999-05-21": the first 4 characters are the year either way.
df["year"] = pd.to_numeric(df["release_date"].astype(str).str[:4], errors="coerce")
broken = df["year"] < FIRST_FILM_YEAR
print("  broken years (before", FIRST_FILM_YEAR, "):", broken.sum())
df.loc[broken, "year"] = None
df["year"] = df["year"].astype("Int64")
df = df.drop(columns=["release_date"])


# ---------- Step 7: filter out rows we don't want ----------
df["plot_words"] = count_words(df["plot"])
df = drop(df, df["title"] != "", "no title")
df = drop(df, df["year"].notna(), "no year")
df = drop(df, df["year"] >= MIN_YEAR, f"before {MIN_YEAR}")
df = drop(df, df["plot_words"] >= MIN_WORDS, f"plot under {MIN_WORDS} words")


# ---------- Step 8: remove duplicates inside this dataset ----------
df["title_key"] = df["title"].apply(make_title_key)

# Same title + same year = the same movie (for example a Telugu and a Tamil version).
# We keep ONE entry with the longest plot, but join the languages, countries
# and genres of all copies, so nothing is lost.
def join_lists(column):
    return sorted({item for items in column for item in items})

joined = df.groupby(["title_key", "year"]).agg(
    languages=("languages", join_lists),
    countries=("countries", join_lists),
    genres=("genres", join_lists),
    other_wiki_ids=("wiki_id", list),
).reset_index()

df = df.sort_values("plot_words", ascending=False)
before = len(df)
df = (df.drop_duplicates(["title_key", "year"], keep="first")
        .drop(columns=["languages", "countries", "genres"])
        .merge(joined, on=["title_key", "year"]))
df["other_wiki_ids"] = [[i for i in ids if i != main]
                        for ids, main in zip(df["other_wiki_ids"], df["wiki_id"])]
print(f"  removed {before - len(df):>6}  (duplicate: same title + year, languages joined)")


# ---------- Add an ID and the source ----------
df["source"] = "cmu"
df["id"] = "cmu_" + df["wiki_id"].astype(str)
df = df.sort_values(["year", "title"]).reset_index(drop=True)

# Same column names as Wikipedia where the meaning is the same
df = df[["id", "title", "title_key", "year", "languages", "countries", "genres",
         "plot", "plot_words", "wiki_id", "other_wiki_ids", "source"]]


# ---------- Step 9: check again ----------
print("\nEnd:", len(df), "movies")
print("Years:", df["year"].min(), "to", df["year"].max())
print("Movies without genre:", (df["genres"].str.len() == 0).sum())
print("Movies without language:", (df["languages"].str.len() == 0).sum())
print("IDs unique:", df["id"].is_unique)
print("\nPlot length (words):\n", df["plot_words"].describe().round(0))
print("\nTop languages:\n", df["languages"].explode().value_counts().head(10))
print("\nTop genres:\n", df["genres"].explode().value_counts().head(10))
print("\nOne cleaned example:\n", df.iloc[0].to_dict())


# ---------- Step 10: save as JSON ----------
save_json(df, OUT)