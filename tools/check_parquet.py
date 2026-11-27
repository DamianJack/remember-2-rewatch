"""
Check that data/movies.parquet holds exactly the same movies as data/merged/movies.json.

Run from the main project folder:
    python tools/check_parquet.py
"""
import json

import pandas as pd

with open("data/merged/movies.json", encoding="utf-8") as f:
    original = json.load(f)
pq = pd.read_parquet("data/movies.parquet")

print("Movies in JSON:   ", len(original))
print("Movies in Parquet:", len(pq))
print("Same columns:     ", sorted(original[0].keys()) == sorted(pq.columns))

# Compare every movie: id, title, year and the three texts must be identical
bad = 0
for m, (_, row) in zip(original, pq.iterrows()):
    if (m["id"], m["title"], m["year"], m["wiki_plot"], m["cmu_plot"], m["tmdb_overview"]) != \
       (row["id"], row["title"], row["year"], row["wiki_plot"], row["cmu_plot"], row["tmdb_overview"]):
        bad += 1
print("Movies that differ:", bad)

# Look at one known movie
hit = pq[pq["title"].str.contains("hubali 2", case=False)]
for _, row in hit.iterrows():
    print(f"\nExample: {row['title']} ({row['year']})  languages={list(row['languages'])}  "
          f"sources={list(row['sources'])}")
    print("   source_ids:", json.loads(row["source_ids"]))

print("\nAll good!" if bad == 0 and len(original) == len(pq) else "\nSomething differs, tell Claude.")