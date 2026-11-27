"""
Save the merged movie list as a compressed Parquet file, small enough for GitHub.

Input:  data/merged/movies.json   (~259 MB, not on GitHub)
Output: data/movies.parquet       (compressed, this one goes on GitHub)

Run from the main project folder:
    python tools/to_parquet.py
"""
import json
import os

import pandas as pd

SRC = "data/merged/movies.json"
OUT = "data/movies.parquet"
GITHUB_LIMIT_MB = 100

with open(SRC, encoding="utf-8") as f:
    movies = json.load(f)
df = pd.DataFrame(movies)

# source_ids is a small dict per movie ({"tmdb": ..., "wikipedia": ...}); store it as text
# so every row has the same shape
df["source_ids"] = df["source_ids"].apply(lambda d: json.dumps(d, ensure_ascii=False))

df.to_parquet(OUT, engine="pyarrow", compression="zstd", compression_level=19, index=False)

size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"Saved {len(df)} movies to {OUT}: {size_mb:.0f} MB "
      f"(JSON was {os.path.getsize(SRC) / 1024 / 1024:.0f} MB)")

# Check: read it back and compare
back = pd.read_parquet(OUT)
print("Read back OK:", len(back) == len(df), "| columns:", len(back.columns))
if size_mb > GITHUB_LIMIT_MB:
    print(f"WARNING: over GitHub's {GITHUB_LIMIT_MB} MB limit, don't push it. Tell me the size.")
else:
    print("Fits on GitHub.")