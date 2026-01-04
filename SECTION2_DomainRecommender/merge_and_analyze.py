
import pandas as pd
import os
import sys

# Paths
base_file = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/streamer_metadata.csv'
fixed_file = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/streamer_metadata_fixed.csv'
final_file = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/streamer_metadata.csv' # Overwrite main file

print("Loading datasets...")

dfs = []
if os.path.exists(base_file):
    df_base = pd.read_csv(base_file)
    dfs.append(df_base)
    print(f"Base file rows: {len(df_base)}")

if os.path.exists(fixed_file):
    try:
        df_fixed = pd.read_csv(fixed_file)
        if len(df_fixed) > 0:
            dfs.append(df_fixed)
            print(f"Fixed file rows: {len(df_fixed)}")
        else:
            print("Fixed file is empty.")
    except Exception as e:
        print(f"Error reading fixed file: {e}")

if not dfs:
    print("No data found!")
    sys.exit(1)

# Combine
combined_df = pd.concat(dfs, ignore_index=True)

# Drop duplicates (keep last - latest scrape might be better, or keep first if base is better? 
# Base was "clean" but "fixed" is rescrape. Assuming rescrape is improvement or filling missing.
# If base had it, it wasn't in rescrape list. If base didn't have it, it was. 
# So overlap should be zero unless I messed up.
combined_df.drop_duplicates(subset=['NAME'], keep='last', inplace=True)

print(f"\nTotal Unique Streamers: {len(combined_df)}")

# Analysis
print("\n=== MISSING DATA ANALYSIS ===")
missing_counts = combined_df.isnull().sum()
print(missing_counts[missing_counts > 0])

# Validity Checks
# Strict: All cols except Rank
cols_strict = [c for c in combined_df.columns if c != 'RANK']
valid_strict = combined_df.dropna(subset=cols_strict)

# Lifetime: User defined [NAME, TOTAL_FOLLOWERS, MOST_STREAMED_GAME, TOTAL_TIME_STREAMED, TOTAL_GAMES_STREAMED]
cols_lifetime = ['NAME', 'TOTAL_FOLLOWERS', 'MOST_STREAMED_GAME', 'TOTAL_TIME_STREAMED', 'TOTAL_GAMES_STREAMED']
valid_lifetime = combined_df.dropna(subset=cols_lifetime)

print("\n=== COMPLETENESS STATS ===")
print(f"Strictly Complete (All fields, ignoring Rank): {len(valid_strict)} ({len(valid_strict)/len(combined_df)*100:.1f}%)")
print(f"Lifetime Complete (Followers, Time, Games, Name): {len(valid_lifetime)} ({len(valid_lifetime)/len(combined_df)*100:.1f}%)")

# Save
print(f"\nSaving merged dataset to {final_file}...")
combined_df.to_csv(final_file, index=False)
print("Done.")
