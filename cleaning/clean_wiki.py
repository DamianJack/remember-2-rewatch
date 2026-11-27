"""
Clean the Wikipedia Movie Plots dataset.

Input:  data/raw/wiki_movie_plots_deduped.csv
Output: data/clean/wiki_clean.json

Run from the main project folder:
    python cleaning/clean_wiki.py
"""
import re

import pandas as pd

from common import (FIRST_FILM_YEAR, clean_text, count_words, drop,
                    make_title_key, save_json)

RAW = "data/raw/wiki_movie_plots_deduped.csv"
OUT = "data/clean/wiki_clean.json"

# Our decisions, kept in one place so they're easy to change
MIN_YEAR = FIRST_FILM_YEAR   # keep all years (we don't know what people will search)
MIN_WORDS = 30               # drop plots that are too short to search


# ---------- Step 1: load ----------
df = pd.read_csv(RAW)
print("Start:", len(df), "movies")


# ---------- Steps 3 + 4: keep the columns we need, with shared names ----------
df = df.rename(columns={
    "Title": "title",
    "Release Year": "year",
    "Origin/Ethnicity": "origin",
    "Director": "director",
    "Cast": "cast",
    "Genre": "genre",
    "Wiki Page": "wiki_url",
    "Plot": "plot",
})
df = df[["title", "year", "origin", "director", "cast", "genre", "wiki_url", "plot"]]


# ---------- Step 5: clean the text ----------
for col in ["title", "origin", "director", "cast", "plot"]:
    df[col] = df[col].apply(clean_text)

# "Unknown" is not a real director
df["director"] = df["director"].replace({"Unknown": "", "unknown": ""})


# Genre: "comedy, drama" -> ["comedy", "drama"];  "unknown" -> []
def clean_genres(text):
    text = clean_text(text).lower()
    if text in ("", "unknown"):
        return []
    parts = re.split(r",|/|;| and ", text)
    return sorted({p.strip() for p in parts if p.strip()})

df["genres"] = df["genre"].apply(clean_genres)
df = df.drop(columns=["genre"])


# Cast: "Tom Hanks, Tim Allen" -> ["Tom Hanks", "Tim Allen"]
def clean_cast(text):
    if text.lower() in ("", "unknown"):
        return []
    return [name.strip() for name in text.split(",") if name.strip()]

df["cast"] = df["cast"].apply(clean_cast)


# ---------- Step 6: one clean year column ----------
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")


# ---------- Step 7: filter out rows we don't want ----------
df["plot_words"] = count_words(df["plot"])
df = drop(df, df["title"] != "", "no title")
df = drop(df, df["year"].notna(), "no year")
df = drop(df, df["year"] >= MIN_YEAR, f"before {MIN_YEAR}")
df = drop(df, df["plot_words"] >= MIN_WORDS, f"plot under {MIN_WORDS} words")


# ---------- Step 8: remove duplicates inside this dataset ----------
df["title_key"] = df["title"].apply(make_title_key)

# Same title + same year + same origin = the same movie. Keep the longest plot.
df = df.sort_values("plot_words", ascending=False)
df = drop(df, ~df.duplicated(["title_key", "year", "origin"]), "duplicate")


# ---------- Step 8b: one Wikipedia page = one movie ----------
# The same page can appear under different years, names or origins.
# We keep ONE entry, and save the other names, years and origins in lists.

# Which row gives the main title? Prefer a title WITHOUT brackets,
# so "Baahubali: The Conclusion" wins over "Baahubali: The Conclusion (Dubbed version)".
# If that's a tie, the earliest year wins.
df["has_brackets"] = df["title"].str.contains(r"\(", regex=True)
df = df.sort_values(["has_brackets", "year"])

# The main TITLE comes from the row chosen above, but the PLOT is the longest one in the group
extra = df.groupby("wiki_url").agg(
    all_titles=("title", lambda t: sorted(set(t))),
    all_years=("year", lambda y: sorted(set(int(v) for v in y))),
    origins=("origin", lambda o: sorted(set(o))),
    longest_plot=("plot", lambda p: max(p, key=len)),
).reset_index()

before = len(df)
df = df.drop_duplicates("wiki_url", keep="first").merge(extra, on="wiki_url")
df["other_titles"] = [[t for t in titles if t != main]
                      for titles, main in zip(df["all_titles"], df["title"])]
df["year"] = [years[0] for years in df["all_years"]]          # main year = earliest
df["other_years"] = [years[1:] for years in df["all_years"]]
df["plot"] = df["longest_plot"]
df["plot_words"] = count_words(df["plot"])
df = df.drop(columns=["origin", "all_titles", "all_years", "has_brackets", "longest_plot"])
print(f"  removed {before - len(df):>6}  (same Wikipedia page)")


# ---------- Add an ID and the source ----------
df["source"] = "wikipedia"
df["id"] = df["wiki_url"]          # the Wikipedia link never changes, so it's a stable ID
df = df.sort_values(["year", "title"]).reset_index(drop=True)

# Put the columns in a readable order
df = df[["id", "title", "other_titles", "title_key", "year", "other_years", "origins",
         "genres", "director", "cast", "plot", "plot_words", "wiki_url", "source"]]


# ---------- Step 9: check again ----------
print("\nEnd:", len(df), "movies")
print("Years:", df["year"].min(), "to", df["year"].max())
print("Movies without genre:", (df["genres"].str.len() == 0).sum())
print("Movies with other titles:", (df["other_titles"].str.len() > 0).sum())
print("Movies with other years:", (df["other_years"].str.len() > 0).sum())
print("Main titles still with brackets:", df["title"].str.contains(r"\(").sum())
print("IDs unique:", df["id"].is_unique)
print("\nTop origins:\n", df["origins"].explode().value_counts().head(10))
print("\nOne cleaned example:\n", df.iloc[0].to_dict())


# ---------- Step 10: save as JSON ----------
save_json(df, OUT)