"""
Clean the TMDB dataset (Full TMDB Movies Dataset, ~1.5 million movies).

Input:  data/raw/TMDB_movie_dataset_v11.csv
Output: data/clean/tmdb_clean.json

Run from the main project folder:
    python cleaning/clean_tmdb.py
"""
from datetime import date

import pandas as pd

from common import (FIRST_FILM_YEAR, clean_text, count_words, drop,
                    make_title_key, save_json)

RAW = "data/raw/TMDB_movie_dataset_v11.csv"
OUT = "data/clean/tmdb_clean.json"

# Our decisions, kept in one place so they're easy to change
MIN_VOTES = 10           # a movie needs this many votes to be kept...
MIN_VOTES_INDIAN = 0    # ...except Indian-language films, which get fewer votes on TMDB
INDIAN_LANGUAGES = ["hi", "te", "ta", "ml", "kn", "bn", "mr", "pa", "gu", "or", "as"]
MIN_WORDS = 10           # TMDB overviews are short, so a lower minimum than the plots (30)

# Overviews that are not a description of the story
PLACEHOLDERS = {
    "mexican feature film", "short film", "plot unavailable", "hong kong movie",
    "direct to video", "nikkatsu roman porno", "no overview found",
    "we don't have an overview translated in english. help us expand our database by adding one",
}

# Language codes -> names, so they match CMU and Wikipedia ("te" -> "Telugu")
LANGUAGE_NAMES = {
    "en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil", "ml": "Malayalam",
    "kn": "Kannada", "bn": "Bengali", "mr": "Marathi", "pa": "Punjabi", "gu": "Gujarati",
    "or": "Odia", "as": "Assamese", "ur": "Urdu", "fr": "French", "es": "Spanish",
    "de": "German", "it": "Italian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "cn": "Cantonese", "pt": "Portuguese", "ru": "Russian", "tr": "Turkish", "ar": "Arabic",
    "fa": "Persian", "nl": "Dutch", "sv": "Swedish", "da": "Danish", "no": "Norwegian",
    "fi": "Finnish", "pl": "Polish", "cs": "Czech", "hu": "Hungarian", "el": "Greek",
    "he": "Hebrew", "th": "Thai", "id": "Indonesian", "tl": "Tagalog", "vi": "Vietnamese",
    "ro": "Romanian", "uk": "Ukrainian", "sr": "Serbian", "hr": "Croatian", "is": "Icelandic",
}

COLUMNS = ["id", "title", "original_title", "release_date", "status", "adult",
           "vote_count", "vote_average", "popularity", "original_language",
           "spoken_languages", "production_countries", "genres", "keywords",
           "overview", "tagline", "imdb_id"]


# ---------- Step 1: load (only the columns we need) ----------
df = pd.read_csv(RAW, usecols=COLUMNS, low_memory=False)
print("Start:", len(df), "movies")


# ---------- Step 7 (part 1): the cheap filters first, so everything after is fast ----------
df = drop(df, df["status"] == "Released", "not released")
df = drop(df, df["adult"].astype(str).str.lower() != "true", "adult")

is_indian = df["original_language"].isin(INDIAN_LANGUAGES)
enough_votes = (df["vote_count"] >= MIN_VOTES) | (is_indian & (df["vote_count"] >= MIN_VOTES_INDIAN))
df = drop(df, enough_votes, f"under {MIN_VOTES} votes ({MIN_VOTES_INDIAN} for Indian languages)")


# ---------- Step 6: one clean year column ----------
release = pd.to_datetime(df["release_date"], errors="coerce")
df["year"] = release.dt.year.astype("Int64")
df = drop(df, df["year"].notna(), "no release date")
df = drop(df, df["year"] >= FIRST_FILM_YEAR, f"before {FIRST_FILM_YEAR}")
df = drop(df, release.loc[df.index] <= pd.Timestamp(date.today()), "release date in the future")
df["release_date"] = release.loc[df.index].dt.strftime("%Y-%m-%d")


# ---------- Step 5: clean the text ----------
for col in ["title", "original_title", "overview", "tagline"]:
    df[col] = df[col].apply(clean_text)


def to_list(text):
    """'Action, Drama' -> ['Action', 'Drama'];  empty -> []"""
    if pd.isna(text) or str(text).strip() == "":
        return []
    return sorted({part.strip() for part in str(text).split(",") if part.strip()})

df["genres"] = df["genres"].apply(lambda t: [g.lower() for g in to_list(t)])
df["keywords"] = df["keywords"].apply(lambda t: [k.lower() for k in to_list(t)])
df["countries"] = df["production_countries"].apply(to_list)

# Languages: the original language (as a name) + the spoken languages
def languages(row):
    names = set(to_list(row["spoken_languages"]))
    code = row["original_language"]
    names.add(LANGUAGE_NAMES.get(code, code))
    return sorted(n for n in names if n)

df["languages"] = df.apply(languages, axis=1)
df["original_language"] = df["original_language"].map(lambda c: LANGUAGE_NAMES.get(c, c))

# Original title as another name, if it's different (e.g. బాహుబలి)
df["other_titles"] = [[o] if o and o != t else []
                      for t, o in zip(df["title"], df["original_title"])]


# ---------- Step 7 (part 2): filters that need the cleaned text ----------
df["overview_words"] = count_words(df["overview"])
df = drop(df, df["title"] != "", "no title")
is_placeholder = df["overview"].str.lower().str.rstrip(".").isin(PLACEHOLDERS)
df = drop(df, ~is_placeholder, "placeholder overview")
df = drop(df, df["overview_words"] >= MIN_WORDS, f"overview under {MIN_WORDS} words")


# ---------- Step 8: remove duplicates inside this dataset ----------
df["title_key"] = df["title"].apply(make_title_key)

# Same title + same year = the same movie. Keep the one with the most votes,
# but join the lists of all copies, so nothing is lost.
def join_lists(column):
    return sorted({item for items in column for item in items})

joined = df.groupby(["title_key", "year"]).agg(
    languages=("languages", join_lists),
    countries=("countries", join_lists),
    genres=("genres", join_lists),
    keywords=("keywords", join_lists),
    other_titles=("other_titles", join_lists),
    other_tmdb_ids=("id", list),
).reset_index()

df = df.sort_values("vote_count", ascending=False)
before = len(df)
df = (df.drop_duplicates(["title_key", "year"], keep="first")
        .drop(columns=["languages", "countries", "genres", "keywords", "other_titles"])
        .merge(joined, on=["title_key", "year"]))
df["other_tmdb_ids"] = [[i for i in ids if i != main]
                        for ids, main in zip(df["other_tmdb_ids"], df["id"])]
print(f"  removed {before - len(df):>6}  (duplicate: same title + year, lists joined)")


# ---------- Add an ID and the source ----------
df["tmdb_id"] = df["id"]
df["id"] = "tmdb_" + df["id"].astype(str)
df["source"] = "tmdb"
df = df.sort_values(["year", "title"]).reset_index(drop=True)

df = df[["id", "title", "other_titles", "title_key", "year", "release_date",
         "original_language", "languages", "countries", "genres", "keywords",
         "overview", "overview_words", "tagline", "popularity", "vote_count",
         "vote_average", "imdb_id", "tmdb_id", "other_tmdb_ids", "source"]]


# ---------- Step 9: check again ----------
print("\nEnd:", len(df), "movies")
print("Years:", df["year"].min(), "to", df["year"].max())
print("Movies after 2017 (not in Wikipedia/CMU):", (df["year"] > 2017).sum())
print("Movies without genre:", (df["genres"].str.len() == 0).sum())
print("Movies without keywords:", (df["keywords"].str.len() == 0).sum())
print("Movies with IMDb ID:", df["imdb_id"].notna().sum())
print("IDs unique:", df["id"].is_unique)
print("\nOverview length (words):\n", df["overview_words"].describe().round(0))
print("\nTop original languages:\n", df["original_language"].value_counts().head(12))
print("\nOne cleaned example:\n", df.iloc[len(df) // 2].to_dict())


# ---------- Step 10: save as JSON ----------
save_json(df, OUT)