
import pandas as pd
import os

csv_file = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/streamer_metadata.csv'
list_file = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/rescrape_list.txt'

try:
    df = pd.read_csv(csv_file)
    original_count = len(df)
    
    # Define "Broken" or "Partial"
    # Broken: Missing TOTAL_FOLLOWERS (The 155)
    # Partial: Missing AVG_VIEWERS_PER_STREAM (The 536)
    
    mask_broken = df['TOTAL_FOLLOWERS'].isnull()
    mask_partial = df['AVG_VIEWERS_PER_STREAM'].isnull()
    
    # Combined mask to rescrape
    mask_rescrape = mask_broken | mask_partial
    
    rescrape_df = df[mask_rescrape]
    keep_df = df[~mask_rescrape]
    
    print(f"Original entries: {original_count}")
    print(f"Entries to keep: {len(keep_df)}")
    print(f"Entries to rescrape: {len(rescrape_df)}")
    
    # Save the list of names to rescrape
    rescrape_names = rescrape_df['NAME'].unique().tolist()
    with open(list_file, 'w') as f:
        for name in rescrape_names:
            f.write(f"{name}\n")
            
    print(f"Saved {len(rescrape_names)} names to {list_file}")
    
    # Update the CSV (Removing bad rows)
    keep_df.to_csv(csv_file, index=False)
    print(f"Updated {csv_file} (Removed {len(rescrape_df)} rows)")
    
except Exception as e:
    print(f"Error: {e}")
