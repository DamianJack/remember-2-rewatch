"""
Look at the CMU Movie Summary Corpus (no changes, only printing).

Run from the main project folder:
    python cleaning/explore_cmu.py
"""
import csv
import json

import pandas as pd

FOLDER = "data/raw/MovieSummaries"

# ---------- Step 1: load the two files ----------
# Neither file has a header row, so we give the column names ourselves.
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

# ---------- Step 2: look ----------
print("Metadata:", meta.shape, "   Plots:", plots.shape)
print("\nEmpty cells in metadata:\n", meta.isna().sum())
print("\nFirst 3 metadata rows:\n", meta.head(3).to_string())

# How well do the two files fit together?
meta_ids, plot_ids = set(meta["wiki_id"]), set(plots["wiki_id"])
print("\nMovies with BOTH plot and metadata:", len(meta_ids & plot_ids))
print("Plots WITHOUT metadata:", len(plot_ids - meta_ids))
print("Metadata WITHOUT plot:", len(meta_ids - plot_ids))
print("Repeated IDs in metadata:", meta["wiki_id"].duplicated().sum(),
      "  in plots:", plots["wiki_id"].duplicated().sum())

# Dates: what formats are there?
print("\nDate examples:", meta["release_date"].dropna().sample(8, random_state=1).tolist()
      if len(meta) >= 8 else meta["release_date"].tolist())
year = pd.to_numeric(meta["release_date"].astype(str).str[:4], errors="coerce")
print("Years:", year.min(), "to", year.max(), "  missing:", year.isna().sum())
print("Movies from 1980 on:", (year >= 1980).sum())

# Languages, countries, genres are stored like {"/m/02h40lc": "English Language"}
def names(text):
    return list(json.loads(text).values())

for col in ["languages", "countries", "genres"]:
    counts = meta[col].apply(names).explode().value_counts()
    print(f"\nTop {col} ({len(counts)} different):\n", counts.head(12))
    print("  movies with none:", (meta[col] == "{}").sum())

# Plots
words = plots["plot"].str.split().str.len()
print("\nPlot length (words):\n", words.describe())

print("\n--- 3 random plots ---")
for p in plots.sample(min(3, len(plots)), random_state=1)["plot"]:
    print("\n", p[:600])