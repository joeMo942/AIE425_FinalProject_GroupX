
import pandas as pd
import sys

csv_file = '/home/yousef/AIE425_FinalProject_GroupX/SECTION2_DomainRecommender/data/streamer_metadata.csv'

try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print("CSV file not found.")
    sys.exit(1)
except pd.errors.EmptyDataError:
    print("CSV file is empty.")
    sys.exit(1)

total_streamers = len(df)
print(f"Total streamers completed: {total_streamers}")

# detailed missing data analysis
missing_report = df.isnull().sum()
print("\nMissing values per column:")
print(missing_report[missing_report > 0])

# Rows with ANY missing data
rows_with_missing = df[df.isnull().any(axis=1)]
count_missing = len(rows_with_missing)

print(f"\nCount of streamers with AT LEAST ONE missing field: {count_missing}")

if count_missing > 0:
    print("\nStreamers with missing data (First 20):")
    print(rows_with_missing['NAME'].head(20).tolist())
    
    # Optional: Group by what is missing
    print("\nBreakdown of missing fields:")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            missing_in_col = df[df[col].isnull()]['NAME'].tolist()
            count = len(missing_in_col)
            print(f" - {col}: {count} missing (e.g., {missing_in_col[:5]})")
