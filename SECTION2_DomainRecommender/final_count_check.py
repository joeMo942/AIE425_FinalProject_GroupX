
import pandas as pd
import os

base_file = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/streamer_metadata.csv'
fixed_file = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/streamer_metadata_fixed.csv'

dfs = []

# Load Base File (cleaned, 1060 rows)
if os.path.exists(base_file):
    df_base = pd.read_csv(base_file)
    dfs.append(df_base)
    print(f"Base file rows: {len(df_base)}")

# Load Fixed File (rescrape output)
if os.path.exists(fixed_file):
    try:
        df_fixed = pd.read_csv(fixed_file)
        # Check if empty
        if len(df_fixed) > 0:
            dfs.append(df_fixed)
            print(f"Fixed file rows (so far): {len(df_fixed)}")
        else:
             print("Fixed file exists but is empty.")
    except Exception as e:
        print(f"Error reading fixed file: {e}")
else:
    print("Fixed file does not exist yet.")

if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Validation Logic: Ignore Rank, check other fields
    cols_to_check = [c for c in combined_df.columns if c != 'RANK']
    
    # 2. Filter for Validity
    valid_df = combined_df.dropna(subset=cols_to_check)
    
    # 3. Filter for Lifetime Validity (Ignore Avg Viewers, Duration, Active Days, Rank)
    # Check only: TOTAL_FOLLOWERS, TOTAL_TIME_STREAMED, TOTAL_GAMES_STREAMED, NAME
    lifetime_cols = ['NAME', 'TOTAL_FOLLOWERS', "MOST_STREAMED_GAME",'TOTAL_TIME_STREAMED', 'TOTAL_GAMES_STREAMED']
    valid_lifetime_df = combined_df.dropna(subset=lifetime_cols)
    
    print("\n--------------------------------")
    print(f"Total Combined Streamers: {len(combined_df)}")
    print(f"STRICT Valid (All stats except Rank): {len(valid_df)}")
    print(f"LIFETIME Valid (Has Followers/Time/Games): {len(valid_lifetime_df)}")
    print("--------------------------------")
    
    # Check how many from the rescrape (684 target) are now valid
    # This assumes we know which ones are new. Roughly: valid_df - base_df
    # But names are unique
    if os.path.exists(base_file):
        base_names = set(df_base['NAME'])
        valid_names = set(valid_df['NAME'])
        newly_fixed = len(valid_names) - len(base_names.intersection(valid_names))
        print(f"New valid records from rescrape: {newly_fixed}")

else:
    print("No data found.")
