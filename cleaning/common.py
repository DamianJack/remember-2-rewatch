"""
Shared helpers for all cleaning scripts (Wikipedia, CMU, TMDB).

Every dataset must clean text and build title keys in EXACTLY the same way,
otherwise the merge step can't match the same movie across sources.
Change something here once, and it changes for all three.
"""
import json
import re
import unicodedata

import pandas as pd
from ftfy import fix_text

# Films started around 1888; any earlier year is a broken date (e.g. "1010")
FIRST_FILM_YEAR = 1888


def clean_text(text):
    """Tidy one piece of text: fix broken letters, remove Wikipedia leftovers, fix spaces."""
    if pd.isna(text):
        return ""
    text = fix_text(str(text))                     # fix broken letters (Ã© -> é)
    text = re.sub(r"\{\{.*?\}\}", "", text)        # remove wiki templates like {{Plot}}
    text = re.sub(r"\[\d+\]", "", text)            # remove [1], [23]
    text = re.sub(r"\[(citation needed|clarification needed|disambiguation needed|when\?|who\?)\]",
                  "", text, flags=re.IGNORECASE)   # remove Wikipedia notes
    text = re.sub(r"\[[a-z]\]", "", text)          # remove footnote letters [a], [b]
    text = re.sub(r"\[(\w+)\]", r"\1", text)       # [his] -> his (keep the word)
    text = re.sub(r"\s+", " ", text)               # many spaces/line breaks -> one space
    text = re.sub(r" ([.,;:!?])", r"\1", text)     # "Orlock ." -> "Orlock."
    return text.strip()


def make_title_key(title):
    """A simple version of a title, for matching: lowercase, no accents, no punctuation."""
    t = unicodedata.normalize("NFKD", title)                  # split é into e + accent
    t = "".join(c for c in t if not unicodedata.combining(c)) # drop the accents
    t = t.lower()
    t = re.sub(r"[^\w ]", "", t)       # remove punctuation, but keep letters from any language
    t = re.sub(r"\s+", " ", t).strip()
    return t or title.lower()          # never return an empty key


def count_words(series):
    """Number of words in each text of a column."""
    return series.str.split().str.len().fillna(0).astype(int)


def drop(df, mask, reason):
    """Keep only the rows where mask is True, and print how many were removed."""
    before = len(df)
    df = df[mask]
    print(f"  removed {before - len(df):>6}  ({reason})")
    return df


def save_json(df, path):
    """Save a table as a readable JSON list (normal slashes, real letters, indented)."""
    records = json.loads(df.to_json(orient="records", force_ascii=False))  # turns numpy types into plain ones
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("\nSaved to", path)

def make_loose_key(title_key):
    """
    A looser version of the title key, for spelling differences in
    Indian (and other) titles written in English letters.
    Only used in the merge, as a second try when the normal key doesn't match.
    """
    k = title_key
    k = k.replace("ee", "i").replace("oo", "u")   # Deewana = Diwana, Phoolon = Phulon
    k = re.sub(r"ie\b", "i", k)                    # Kabhie = Kabhi (at the end of a word)
    k = k.replace("iya", "ia")                     # Dulhaniya = Dulhania
    k = re.sub(r"(\w)\1+", r"\1", k)               # double letters -> one: Baahubali = Bahubali
    k = re.sub(r"\s*\b\d\b\s*", " ", k).strip()     # "bahubali 2 the conclusion" = "bahubali the conclusion"   
    return k