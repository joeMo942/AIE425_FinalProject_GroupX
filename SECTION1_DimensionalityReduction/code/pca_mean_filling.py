# PCA with Mean Filling
import os
import pandas as pd

# Define results directory path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CODE_DIR, '..', 'results')
PLOTS_DIR = os.path.join(CODE_DIR, '..', 'plots')

# =============================================================================
# Step 1: Load r_u (user average ratings) and r_i (item average ratings)
# =============================================================================

from utils import (get_user_avg_ratings, get_item_avg_ratings, 
                   get_target_users, get_target_items,
                   get_preprocessed_dataset, create_user_item_matrix,
                   mean_fill_matrix)

# Load user average ratings (r_u)
r_u = get_user_avg_ratings()
print("User Average Ratings (r_u):")
print(r_u.head())
print(f"Shape: {r_u.shape}\n")

# Load item average ratings (r_i)
r_i = get_item_avg_ratings()
print("Item Average Ratings (r_i):")
print(r_i.head())
print(f"Shape: {r_i.shape}")

# Save Step 1 results
print("\n[Saving Step 1 results...]")
r_u.to_csv(os.path.join(RESULTS_DIR, 'step1_r_u.csv'), index=False)
r_i.to_csv(os.path.join(RESULTS_DIR, 'step1_r_i.csv'), index=False)
print("Saved r_u and r_i to results folder.")

# =============================================================================
# Step 2: Load target users and target items
# =============================================================================

# Load target users
target_users = get_target_users()
print("\nTarget Users:")
print(target_users)

# Load target items
target_items = get_target_items()
print("\nTarget Items:")
print(target_items)

# Save Step 2 results
print("\n[Saving Step 2 results...]")
targets_df = pd.DataFrame({
    'Target_Users': pd.Series(target_users),
    'Target_Items': pd.Series(target_items)
})
targets_df.to_csv(os.path.join(RESULTS_DIR, 'step2_targets.csv'), index=False)
print("Saved target users and items to results folder.")

# =============================================================================
# Step 3: Mean-Filling Method - Replace missing ratings with column means
# =============================================================================

# Load the preprocessed dataset
df = get_preprocessed_dataset()
print(f"\nPreprocessed Dataset Shape: {df.shape}")

# Filter dataset to only include target items (I1 and I2) for memory efficiency
print("\nFiltering dataset to target items only...")
df_target_items = df[df['item'].isin(target_items)]
print(f"Filtered Dataset Shape: {df_target_items.shape}")

# Create user-item rating matrix for target items only
print("\nCreating User-Item Rating Matrix for Target Items...")
rating_matrix_target = create_user_item_matrix(df_target_items)
print(f"Rating Matrix Shape: {rating_matrix_target.shape}")
print(f"(Rows = Users who rated target items, Columns = Target Items)")

# Show missing values before filling for target items
total_entries = rating_matrix_target.shape[0] * rating_matrix_target.shape[1]
missing_before = rating_matrix_target.isna().sum().sum()
print(f"\nMissing values before filling: {missing_before:,} ({missing_before/total_entries*100:.2f}%)")

# Calculate mean for each target item (column)
print("\n--- Mean Values for Target Items (I1 and I2) ---")
item_means = {}
for i, item_id in enumerate(target_items, 1):
    item_mean = rating_matrix_target[item_id].mean()
    item_means[item_id] = item_mean
    print(f"I{i} (Item {item_id}): Mean = {item_mean:.6f}")

# Apply mean-filling: Replace missing entries with column means
print("\nApplying Mean-Filling Method...")
rating_matrix_filled = mean_fill_matrix(rating_matrix_target)

# Verify no missing values after filling
missing_after = rating_matrix_filled.isna().sum().sum()
print(f"Missing values after filling: {missing_after}")

# Show sample of original vs filled matrix for comparison
print("\n--- Original Matrix (with missing values) ---")
print(rating_matrix_target.head(10))

print("\n--- Filled Matrix (missing values replaced with mean) ---")
print(rating_matrix_filled.head(10))

# Save Step 3 results
print("\n[Saving Step 3 results...]")
rating_matrix_target.to_csv(os.path.join(RESULTS_DIR, 'step3_original_matrix.csv'))
rating_matrix_filled.to_csv(os.path.join(RESULTS_DIR, 'step3_mean_filled_matrix.csv'))
item_means_df = pd.DataFrame([
    {'Target': 'I1', 'Item_ID': target_items[0], 'Mean': item_means[target_items[0]]},
    {'Target': 'I2', 'Item_ID': target_items[1], 'Mean': item_means[target_items[1]]}
])
item_means_df.to_csv(os.path.join(RESULTS_DIR, 'step3_item_means.csv'), index=False)
print("Saved original matrix, filled matrix, and item means to results folder.")

# =============================================================================
# Step 4: Calculate average rating for each item (using r_i)
# =============================================================================

print("\n" + "="*70)
print("Step 4: Calculate Average Rating for Each Item")
print("="*70)

# We already have r_i loaded, which contains item means
# Display means for target items
print("\nItem means for target items (from r_i):")
for i, item_id in enumerate(target_items, 1):
    item_mean = r_i[r_i['item'] == item_id]['r_i_bar'].values[0]
    print(f"I{i} (Item {item_id}): Mean = {item_mean:.6f}")

# Save Step 4 results (r_i is already item means)
print("\n[Saving Step 4 results...]")
r_i.to_csv(os.path.join(RESULTS_DIR, 'step4_all_item_means.csv'), index=False)
print("Saved all item means to results folder.")

# =============================================================================
# Step 5: Calculate centered ratings (actual - mean) for ALL items
# =============================================================================

print("\n" + "="*70)
print("Step 5: Calculate Centered Ratings (actual - mean) for ALL Items")
print("="*70)

from utils import compute_centered_ratings

# Compute centered ratings for ALL items (entire dataset)
df_centered = compute_centered_ratings(df, r_i)
print(f"\nCentered Ratings DataFrame Shape: {df_centered.shape}")
print("\nSample of centered ratings:")
print(df_centered.head(10))

# Show centered ratings for target items
print(f"\n--- Centered ratings sample for target items (I1={target_items[0]}, I2={target_items[1]}) ---")
df_centered_targets = df_centered[df_centered['item'].isin(target_items)]
print(df_centered_targets.head(10))

# Save Step 5 results
print("\n[Saving Step 5 results...]")
df_centered.to_csv(os.path.join(RESULTS_DIR, 'step5_centered_ratings.csv'), index=False)
print(f"Saved centered ratings for ALL {df_centered.shape[0]:,} ratings to results folder.")

# Step 6: Compute Covariance Matrix for ALL Items (Memory-Efficient)
# =============================================================================

print("\n" + "="*70)
print("Step 6: Compute Covariance Matrix for ALL Items")
print("="*70)

from utils import compute_covariance_matrix_efficient

# Get total number of users (N)
total_users = r_u.shape[0]
print(f"\nTotal number of users (N): {total_users:,}")

# Get all unique items from the dataset
all_items = sorted(r_i['item'].tolist())
print(f"Total number of items: {len(all_items):,}")

# Compute covariance matrix for ALL items
print("\nComputing covariance matrix for ALL items...")
print("(Using memory-efficient algorithm: only users who rated BOTH items contribute)")
print("(Dividing by N-1 for sample covariance)")
print("This may take a few minutes...")

cov_matrix = compute_covariance_matrix_efficient(
    df, all_items, r_i, total_users, show_progress=True
)

print(f"\n--- Covariance Matrix Shape ---")
print(f"Shape: {cov_matrix.shape} ({len(all_items)} x {len(all_items)})")

# Show covariance for target items specifically
print(f"\n--- Covariance Values for Target Items (I1={target_items[0]}, I2={target_items[1]}) ---")
print(f"Var(I1): {cov_matrix.loc[target_items[0], target_items[0]]:.6f}")
print(f"Var(I2): {cov_matrix.loc[target_items[1], target_items[1]]:.6f}")
print(f"Cov(I1, I2): {cov_matrix.loc[target_items[0], target_items[1]]:.6f}")

# Show a sample of the covariance matrix (first 5x5)
print("\n--- Sample of Covariance Matrix (first 5x5) ---")
print(cov_matrix.iloc[:5, :5])

# Save Step 6 results
print("\n[Saving Step 6 results...]")
cov_matrix.to_csv(os.path.join(RESULTS_DIR, 'step6_covariance_matrix.csv'))
print(f"Saved full covariance matrix ({len(all_items)}x{len(all_items)}) to results folder.")

# =============================================================================
# Step 7: Eigenvalue Decomposition - Determine Top 5 and Top 10 Principal Components
# =============================================================================

print("\n" + "="*70)
print("Step 7: PCA Eigenvalue Decomposition")
print("="*70)

import numpy as np
from numpy.linalg import eigh

# Convert covariance matrix to numpy array for eigendecomposition
cov_array = cov_matrix.values

print("\nComputing eigenvalues and eigenvectors of the covariance matrix...")
print(f"Covariance matrix shape: {cov_array.shape}")

# Compute eigenvalues and eigenvectors
# eigh is for symmetric matrices (covariance is symmetric)
eigenvalues, eigenvectors = eigh(cov_array)

# Sort eigenvalues and eigenvectors in descending order
# (eigh returns in ascending order)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

print(f"\n--- Eigenvalues (Top 10) ---")
for i in range(10):
    print(f"  lambda_{i+1} = {eigenvalues[i]:.6f}")

# Total variance
total_variance = np.sum(eigenvalues)
print(f"\nTotal variance (sum of all eigenvalues): {total_variance:.6f}")

# Variance explained by top-k PCs
print(f"\n--- Variance Explained ---")
for k in [5, 10]:
    var_explained = np.sum(eigenvalues[:k]) / total_variance * 100
    print(f"  Top-{k} PCs: {var_explained:.2f}%")

# Top 5 Principal Components (eigenvectors)
print(f"\n--- Top 5 Principal Components (Eigenvectors) ---")
top5_eigenvectors = eigenvectors[:, :5]
print(f"Shape: {top5_eigenvectors.shape} (n_items x 5)")

# Top 10 Principal Components (eigenvectors)
print(f"\n--- Top 10 Principal Components (Eigenvectors) ---")
top10_eigenvectors = eigenvectors[:, :10]
print(f"Shape: {top10_eigenvectors.shape} (n_items x 10)")

# Create projection matrix W for top-5 (this will be used in Step 8)
W_top5 = top5_eigenvectors
W_top10 = top10_eigenvectors

# Save Step 7 results
print("\n[Saving Step 7 results...]")

# Save eigenvalues
eigenvalues_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(len(eigenvalues))],
    'Eigenvalue': eigenvalues,
    'Variance_Explained_Pct': (eigenvalues / total_variance) * 100,
    'Cumulative_Variance_Pct': np.cumsum(eigenvalues / total_variance) * 100
})
eigenvalues_df.to_csv(os.path.join(RESULTS_DIR, 'step7_eigenvalues.csv'), index=False)
print("Saved eigenvalues to results folder.")

# Save top-5 eigenvectors (projection matrix W)
W_top5_df = pd.DataFrame(W_top5, index=all_items, columns=[f'PC{i+1}' for i in range(5)])
W_top5_df.index.name = 'item'
W_top5_df.to_csv(os.path.join(RESULTS_DIR, 'step7_top5_eigenvectors.csv'))
print("Saved top-5 eigenvectors (projection matrix W) to results folder.")

# Save top-10 eigenvectors
W_top10_df = pd.DataFrame(W_top10, index=all_items, columns=[f'PC{i+1}' for i in range(10)])
W_top10_df.index.name = 'item'
W_top10_df.to_csv(os.path.join(RESULTS_DIR, 'step7_top10_eigenvectors.csv'))
print("Saved top-10 eigenvectors to results folder.")

# --- Determine Top 5 and Top 10 Peers for Each Target Item ---
print("\n--- Top Peers for Target Items (based on covariance) ---")

# Function to get top peers for an item based on covariance
def get_top_peers(cov_matrix, item_id, n_peers):
    """Get top n peers for an item based on covariance values."""
    cov_values = cov_matrix.loc[item_id].copy()
    cov_values = cov_values.drop(item_id)  # Remove self
    top_peers = cov_values.sort_values(ascending=False).head(n_peers)
    return top_peers

# Find top peers for each target item
top_peers_results = {}
for i, item_id in enumerate(target_items, 1):
    print(f"\n--- Target Item I{i} (Item {item_id}) ---")
    
    # Top 5 peers
    top5 = get_top_peers(cov_matrix, item_id, 5)
    print(f"\nTop 5 Peers:")
    for rank, (peer_id, cov_val) in enumerate(top5.items(), 1):
        print(f"  {rank}. Item {peer_id}: Cov = {cov_val:.6f}")
    
    # Top 10 peers
    top10 = get_top_peers(cov_matrix, item_id, 10)
    print(f"\nTop 10 Peers:")
    for rank, (peer_id, cov_val) in enumerate(top10.items(), 1):
        print(f"  {rank}. Item {peer_id}: Cov = {cov_val:.6f}")
    
    top_peers_results[f'I{i}'] = {
        'item_id': item_id,
        'top5_peers': top5.index.tolist(),
        'top10_peers': top10.index.tolist()
    }

# Save top peers for target items
top_peers_data = []
for i, item_id in enumerate(target_items, 1):
    top10 = get_top_peers(cov_matrix, item_id, 10)
    for rank, (peer_id, cov_val) in enumerate(top10.items(), 1):
        top_peers_data.append({
            'Target_Item': f'I{i}',
            'Target_Item_ID': item_id,
            'Rank': rank,
            'Peer_Item_ID': peer_id,
            'Covariance': cov_val,
            'Is_Top5': rank <= 5
        })

top_peers_df = pd.DataFrame(top_peers_data)
top_peers_df.to_csv(os.path.join(RESULTS_DIR, 'step7_target_item_peers.csv'), index=False)
print("\nSaved top peers for I1 and I2 to results folder.")

print("\n" + "="*70)
print("[SUCCESS] All steps completed! (Steps 1-7)")
print("="*70)

