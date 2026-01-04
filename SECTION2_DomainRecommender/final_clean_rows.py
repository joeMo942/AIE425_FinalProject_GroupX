
import pandas as pd
import sys

csv_file = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/streamer_metadata.csv'

try:
    df = pd.read_csv(csv_file)
    original_count = len(df)
    
    # Critical columns that MUST be present
    # We allow Rank/Avg Viewers to be missing (inactive users), but Name/Followers/Time must exist.
    critical_cols = ['NAME', 'TOTAL_FOLLOWERS', 'TOTAL_TIME_STREAMED', 'TOTAL_GAMES_STREAMED']
    
    # Check for rows missing ANY of these
    missing_rows = df[df[critical_cols].isnull().any(axis=1)]
    missing_count = len(missing_rows)
    
    if missing_count > 0:
        print(f"Found {missing_count} rows with missing critical data. Dropping them...")
        df_clean = df.dropna(subset=critical_cols)
        df_clean.to_csv(csv_file, index=False)
        print(f"Saved cleaned dataset: {len(df_clean)} rows.")
        print(f"Removed: {missing_count} rows.")
    else:
        print("No missing critical data found. Dataset is already clean.")
        print(f"Total rows: {len(df)}")

except Exception as e:
    print(f"Error: {e}")
