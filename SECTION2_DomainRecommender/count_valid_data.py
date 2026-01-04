
import pandas as pd

csv_file = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/streamer_metadata.csv'

try:
    df = pd.read_csv(csv_file)
    
    # Columns to check (All columns except RANK)
    # Get all columns
    cols_to_check = [c for c in df.columns if c != 'RANK']
    
    # Count rows with NO missing values in these columns
    valid_df = df.dropna(subset=cols_to_check)
    
    # Also count specifically those missing TOTAL_FOLLOWERS (the 155 group)
    broken_df = df[df['TOTAL_FOLLOWERS'].isnull()]
    
    print(f"Total rows in CSV: {len(df)}")
    print(f"Streamers with COMPLETE valid data (ignoring RANK): {len(valid_df)}")
    print(f"Streamers with MISSING critical stats (needs rescrape): {len(broken_df)}")
    
    # Optional: Logic to separate them
    # valid_df.to_csv('valid_streamers.csv', index=False)
    # broken_df['NAME'].to_csv('rescrape_list.txt', index=False, header=False)
    
except Exception as e:
    print(f"Error: {e}")
