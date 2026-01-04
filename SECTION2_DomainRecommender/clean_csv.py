
import pandas as pd
import sys

csv_file = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/streamer_metadata.csv'

try:
    df = pd.read_csv(csv_file)
    if 'TOTAL_VIEWS' in df.columns:
        print(f"Dropping TOTAL_VIEWS column...")
        df.drop(columns=['TOTAL_VIEWS'], inplace=True)
        df.to_csv(csv_file, index=False)
        print("Column dropped and file saved.")
    else:
        print("TOTAL_VIEWS column not found.")
        
    # Also output list of streamers with missing RANK
    missing_rank = df[df['RANK'].isnull()]['NAME'].tolist()
    print(f"\nStreamers with missing RANK ({len(missing_rank)}):")
    with open('missing_rank_streamers.txt', 'w') as f:
        for s in missing_rank:
            f.write(s + "\n")
    print("Saved missing rank list to missing_rank_streamers.txt")
    
except Exception as e:
    print(f"Error: {e}")
