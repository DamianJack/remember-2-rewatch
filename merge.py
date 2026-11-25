"""
Merge the three clean datasets into ONE movie list (one movie = one entry).

Input:  data/clean/wiki_clean.json, data/clean/cmu_clean.json, data/clean/tmdb_clean.json
Output: data/merged/movies.json

Run from the main project folder:
    python merge.py

How we decide "same movie":
  1. same title key (main title OR any other title)
  2. year the same, or 1 year apart (release dates differ by country)
     - very short titles ("up", "11") must have the SAME year, to avoid wrong pairs
  3. if both have language info, they must share at least one language
     (keeps remakes apart, e.g. Drishyam Malayalam vs Drishyam Hindi)
     - exception: two Indian languages in the SAME year count as shared (dubbed versions)
  4. one entry can hold at most ONE row from each source
"""
import json
import os
from collections import defaultdict

CLEAN = "data/clean"
OUT_DIR = "data/merged"
OUT = f"{OUT_DIR}/movies.json"

SHORT_TITLE = 3   # title keys this short (or shorter) need the exact same year

# Wikipedia "origins" that tell us the language
ORIGIN_TO_LANGUAGE = {
    "American": "English", "British": "English", "Australian": "English",
    "Canadian": "English", "Irish": "English", "New Zealand": "English",
    "Bollywood": "Hindi", "Tamil": "Tamil", "Telugu": "Telugu", "Malayalam": "Malayalam",
    "Kannada": "Kannada", "Bengali": "Bengali", "Marathi": "Marathi", "Punjabi": "Punjabi",
    "Assamese": "Assamese", "Japanese": "Japanese", "South_Korean": "Korean",
    "Chinese": "Chinese", "Hong Kong": "Chinese", "Russian": "Russian", "Turkish": "Turkish",
    "Egyptian": "Arabic", "Filipino": "Tagalog", "Malaysian": "Malay",
}
INDIAN_LANGUAGES = {"Hindi", "Telugu", "Tamil", "Malayalam", "Kannada", "Bengali",
                    "Marathi", "Punjabi", "Gujarati", "Odia", "Assamese", "Urdu"}

# Different names for the same language in different sources
SAME_LANGUAGE = {
    "Standard Mandarin": "Chinese", "Mandarin": "Chinese", "Cantonese": "Chinese",
    "Standard Cantonese": "Chinese", "普通话": "Chinese", "广州话 / 廣州話": "Chinese",
}


def load(name):
    with open(f"{CLEAN}/{name}", encoding="utf-8") as f:
        return json.load(f)


def norm_lang(name):
    return SAME_LANGUAGE.get(name, name)


# ---------- Step 1: load and put every row into the same shape ----------
rows = []   # each row: source, data, keys (all title keys), year, langs

def add(source, movie, title_keys, langs):
    rows.append({
        "source": source,
        "movie": movie,
        "keys": {k for k in title_keys if k},
        "loose_keys": {make_loose_key(k) for k in title_keys if k},
        "year": movie["year"],
        "langs": {norm_lang(l) for l in langs if l},
    })

# title_key for other titles, made the same way as in cleaning
import sys
sys.path.insert(0, "cleaning")
from common import make_loose_key, make_title_key  # noqa: E402

for m in load("wiki_clean.json"):
    keys = [m["title_key"]] + [make_title_key(t) for t in m["other_titles"]]
    langs = [ORIGIN_TO_LANGUAGE[o] for o in m["origins"] if o in ORIGIN_TO_LANGUAGE]
    add("wikipedia", m, keys, langs)

for m in load("cmu_clean.json"):
    add("cmu", m, [m["title_key"]], m["languages"])

for m in load("tmdb_clean.json"):
    keys = [m["title_key"]] + [make_title_key(t) for t in m["other_titles"]]
    # spoken languages can be long lists; the original language is the safest signal
    add("tmdb", m, keys, [m["original_language"]] + m["languages"])

print("Rows loaded:", len(rows),
      {s: sum(r["source"] == s for r in rows) for s in ("wikipedia", "cmu", "tmdb")})


# ---------- Step 2: find pairs that are the same movie ----------
# Index: (title key, year) -> row numbers
index = defaultdict(list)
for i, r in enumerate(rows):
    for k in r["keys"]:
        index[(k, r["year"])].append(i)

# A second index with the loose keys (for spelling differences like Baahubali / Bahubali)
loose_index = defaultdict(list)
for i, r in enumerate(rows):
    for k in r["loose_keys"]:
        loose_index[(k, r["year"])].append(i)

# "Union-find": every row starts as its own group; matching rows are joined.
parent = list(range(len(rows)))
group_sources = [{r["source"]} for r in rows]
group_langs = [set(r["langs"]) for r in rows]

def find(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

def languages_ok(la, lb, same_year):
    """Rule 3: both rows must share a language (if both have language info).
    Exception: Indian films are often dubbed into other Indian languages and released
    in the SAME year (Baahubali: Telugu original, Tamil + Malayalam dubs).
    So in the same year, any two Indian languages count as a match.
    (Not for 1 year apart: that is often a remake, e.g. Pokiri 2006 -> Pokkiri 2007.)"""
    if not la or not lb or (la & lb):
        return True
    return same_year and bool(la & INDIAN_LANGUAGES) and bool(lb & INDIAN_LANGUAGES)

def try_join(a, b, same_year):
    ga, gb = find(a), find(b)
    if ga == gb:
        return False
    if group_sources[ga] & group_sources[gb]:          # rule 4: one row per source
        return False
    if not languages_ok(group_langs[ga], group_langs[gb], same_year):   # rule 3
        return False
    parent[gb] = ga
    group_sources[ga] |= group_sources[gb]
    group_langs[ga] |= group_langs[gb]
    return True

# Most reliable first: exact title + same year, then exact title + 1 year apart,
# then the loose spelling (same year, then 1 year apart)
joins = {"same year": 0, "1 year apart": 0, "loose spelling, same year": 0, "loose spelling, 1 year apart": 0}
passes = [
    ((0,), "same year", "keys", index),
    ((-1, 1), "1 year apart", "keys", index),
    ((0,), "loose spelling, same year", "loose_keys", loose_index),
    ((-1, 1), "loose spelling, 1 year apart", "loose_keys", loose_index),
]
for offsets, label, key_field, idx in passes:
    for i, r in enumerate(rows):
        for k in r[key_field]:
            if offsets != (0,) and len(k) <= SHORT_TITLE:
                continue                                # rule 2: short titles need exact year
            for off in offsets:
                for j in idx.get((k, r["year"] + off), []):
                    if j != i and try_join(i, j, same_year=(off == 0)):
                        joins[label] += 1
print("Joins:", joins)


# ---------- Step 3: build one entry per group ----------
groups = defaultdict(list)
for i in range(len(rows)):
    groups[find(i)].append(rows[i])

def first(values):
    for v in values:
        if v not in (None, "", [], {}):
            return v
    return None

def union(*lists):
    out = []
    for lst in lists:
        for v in lst or []:
            if v and v not in out:
                out.append(v)
    return sorted(out, key=str)

merged = []
for members in groups.values():
    by = {r["source"]: r["movie"] for r in members}
    w, c, t = by.get("wikipedia", {}), by.get("cmu", {}), by.get("tmdb", {})

    # Main values: TMDB first (current English title, release year), then Wikipedia, then CMU
    title = first([t.get("title"), w.get("title"), c.get("title")])
    year = first([t.get("year"), w.get("year"), c.get("year")])
    all_titles = union([w.get("title"), c.get("title"), t.get("title")],
                       w.get("other_titles"), t.get("other_titles"))
    all_years = union([w.get("year"), c.get("year"), t.get("year")], w.get("other_years"))
    wiki_langs = [ORIGIN_TO_LANGUAGE[o] for o in w.get("origins", []) if o in ORIGIN_TO_LANGUAGE]

    entry = {
        "id": first([t.get("id"), w.get("id"), c.get("id")]),
        "title": title,
        "other_titles": [x for x in all_titles if x != title],
        "year": year,
        "other_years": [y for y in all_years if y != year],
        "languages": union([norm_lang(l) for l in c.get("languages", []) + t.get("languages", []) + wiki_langs]),
        "original_language": t.get("original_language", ""),
        "origins": w.get("origins", []),
        "countries": union(c.get("countries"), t.get("countries")),
        "genres": union(w.get("genres"), c.get("genres"), t.get("genres")),
        "keywords": t.get("keywords", []),
        "director": w.get("director", ""),
        "cast": w.get("cast", []),
        # All texts side by side: more chances for search to match
        "wiki_plot": w.get("plot", ""),
        "cmu_plot": c.get("plot", ""),
        "tmdb_overview": t.get("overview", ""),
        "tagline": t.get("tagline", ""),
        "popularity": t.get("popularity"),
        "vote_count": t.get("vote_count"),
        "vote_average": t.get("vote_average"),
        "imdb_id": t.get("imdb_id"),
        "sources": sorted(by.keys()),
        "source_ids": {s: m["id"] for s, m in by.items()},
    }
    merged.append(entry)

merged.sort(key=lambda m: (m["year"], m["title"]))


# ---------- Step 4: check ----------
n = len(merged)
ids = [m["id"] for m in merged]
print("\nMerged movies:", n, "(from", len(rows), "rows)")
print("IDs unique:", len(set(ids)) == n)
combo = defaultdict(int)
for m in merged:
    combo[" + ".join(m["sources"])] += 1
print("\nWhich sources each movie came from:")
for k, v in sorted(combo.items(), key=lambda x: -x[1]):
    print(f"  {v:>7}  {k}")
print("\nWithout any genre:", sum(not m["genres"] for m in merged))
print("Without a long plot (only TMDB overview):", sum(not m["wiki_plot"] and not m["cmu_plot"] for m in merged))

for name in ["Inception", "Baahubali: The Beginning", "Bahubali: The Beginning", "Titanic", "Lagaan"]:
    hits = [m for m in merged if m["title"] == name]
    for m in hits[:2]:
        print(f"\nExample: {m['title']} ({m['year']}) sources={m['sources']} languages={m['languages']}")


# ---------- Step 5: save ----------
os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)
print("\nSaved to", OUT)