import pandas as pd

# Step 1: load
df = pd.read_csv("data/raw/wiki_movie_plots_deduped.csv")

# Step 2: look (no changes yet)
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nEmpty cells per column:\n", df.isna().sum())
print("\nFirst 3 rows:\n", df.head(3))
print("\nTop 15 origins:\n", df["Origin/Ethnicity"].value_counts().head(15))
print("\nTop 15 genres:\n", df["Genre"].value_counts().head(15))
print("\nPlot length (words):\n", df["Plot"].str.split().str.len().describe())
print("\nYears:", df["Release Year"].min(), "to", df["Release Year"].max())

print("\n--- 3 random plots ---")
for plot in df.sample(3, random_state=1)["Plot"]:
    print("\n", plot[:600])