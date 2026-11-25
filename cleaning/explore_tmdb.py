"""
Look at the TMDB dataset (no changes, only printing).

The file is big (~680 MB, about 1 million movies), so we only load the columns we need.

Run from the main project folder:
    python cleaning/explore_tmdb.py
"""
import pandas as pd

RAW = "data/raw/TMDB_movie_dataset_v11.csv"

COLUMNS = ["id", "title", "original_title", "release_date", "status", "adult",
           "vote_count", "vote_average", "popularity", "original_language",
           "spoken_languages", "production_countries", "genres", "keywords",
           "overview", "tagline", "imdb_id"]

# ---------- Step 1: load ----------
df = pd.read_csv(RAW, usecols=COLUMNS, low_memory=False)

# ---------- Step 2: look ----------
print("Shape:", df.shape)
print("\nEmpty cells per column:\n", df.isna().sum())
print("\nFirst 3 rows:\n", df.head(3).T)

print("\nStatus:\n", df["status"].value_counts())
print("\nAdult:\n", df["adult"].value_counts())

year = pd.to_numeric(df["release_date"].astype(str).str[:4], errors="coerce")
print("\nYears:", year.min(), "to", year.max(), "  missing:", year.isna().sum())
print("Movies per period:\n", pd.cut(year, [1800, 1980, 2000, 2010, 2017, 2020, 2030]).value_counts().sort_index())

# Most of the 1 million are tiny unknown films. How many have enough votes?
print("\nMovies with at least N votes:")
for n in [0, 1, 5, 10, 20, 50, 100, 500]:
    print(f"  {n:>4}+ votes: {(df['vote_count'] >= n).sum():>8}")

print("\nTop original languages:\n", df["original_language"].value_counts().head(15))
print("\nIndian languages (votes 5+):")
for lang in ["hi", "te", "ta", "ml", "kn", "bn", "mr"]:
    print(f"  {lang}: {((df['original_language'] == lang) & (df['vote_count'] >= 5)).sum()}")

# Overviews: the short text we would search
words = df["overview"].fillna("").str.split().str.len()
print("\nOverview length (words):\n", words.describe())
print("Empty overviews:", (words == 0).sum())
print("Most common overviews (placeholders?):\n", df["overview"].value_counts().head(8))

print("\nGenre examples:", df["genres"].dropna().head(3).tolist())
print("Keyword examples:", df["keywords"].dropna().head(3).tolist())
print("Has IMDb ID:", df["imdb_id"].notna().sum())

print("\n--- 3 random overviews (movies with 50+ votes) ---")
known = df[df["vote_count"] >= 50]
for _, row in known.sample(min(3, len(known)), random_state=1).iterrows():
    print(f"\n {row['title']} ({str(row['release_date'])[:4]}, {row['original_language']}):\n {row['overview']}")