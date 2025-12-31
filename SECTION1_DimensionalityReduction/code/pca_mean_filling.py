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

# =============================================================================
# Step 8 & 10: Project Users into Reduced Latent Space (Top-5 and Top-10)
# =============================================================================

print("\n" + "="*70)
print("Step 8 & 10: Project Users into Reduced Latent Space")
print("="*70)

# Formula: UserVector_u = R'_u × W_k
# Optimization: Only iterate over items the user actually rated (missing = 0)
# UserVector_u[dim] = Sum_{j in Observed} (R_{u,j} - μ_j) × W_{j, dim}

# Get item means as a dictionary for fast lookup
item_means_dict = r_i.set_index('item')['r_i_bar'].to_dict()

# Get all users from the dataset
all_users = sorted(df['user'].unique().tolist())
print(f"\nTotal users to project: {len(all_users):,}")

# Create a mapping: user -> {item: rating}
print("Building user-item rating lookup...")
user_ratings = df.groupby('user').apply(
    lambda x: dict(zip(x['item'], x['rating']))
).to_dict()

def project_user(user_id, W, all_items, item_means_dict, user_ratings):
    """
    Project a single user into the reduced latent space.
    
    Formula: UserVector_u[dim] = Sum_{j in Observed} (R_{u,j} - μ_j) × W_{j, dim}
    
    Args:
        user_id: The user to project
        W: The projection matrix (n_items x k)
        all_items: List of all items (defines row order in W)
        item_means_dict: Dictionary {item_id: mean}
        user_ratings: Dictionary {user_id: {item_id: rating}}
    
    Returns:
        np.array: User vector of size k
    """
    k = W.shape[1]  # Number of dimensions (5 or 10)
    user_vector = np.zeros(k)
    
    if user_id not in user_ratings:
        return user_vector
    
    # Only iterate over items the user actually rated
    for item_id, rating in user_ratings[user_id].items():
        if item_id in item_means_dict:
            # Get item index in W
            item_idx = all_items.index(item_id)
            # Centered rating: R_{u,j} - μ_j
            centered_rating = rating - item_means_dict[item_id]
            # Add contribution to each dimension
            user_vector += centered_rating * W[item_idx, :]
    
    return user_vector

# Project all users using Top-5 PCs
print("\nProjecting all users using Top-5 PCs...")
user_vectors_top5 = {}
for i, user_id in enumerate(all_users):
    user_vectors_top5[user_id] = project_user(
        user_id, W_top5, all_items, item_means_dict, user_ratings
    )
    if (i + 1) % 50000 == 0:
        print(f"  Projected {i+1:,} users...")
print(f"Projected all {len(all_users):,} users to 5-dimensional space.")

# Project all users using Top-10 PCs
print("\nProjecting all users using Top-10 PCs...")
user_vectors_top10 = {}
for i, user_id in enumerate(all_users):
    user_vectors_top10[user_id] = project_user(
        user_id, W_top10, all_items, item_means_dict, user_ratings
    )
    if (i + 1) % 50000 == 0:
        print(f"  Projected {i+1:,} users...")
print(f"Projected all {len(all_users):,} users to 10-dimensional space.")

# Show sample projections for target users
print("\n--- Sample User Projections (Target Users) ---")
for i, user_id in enumerate(target_users, 1):
    print(f"\nTarget User U{i} (User {user_id}):")
    print(f"  Top-5 Vector:  {user_vectors_top5[user_id]}")
    print(f"  Top-10 Vector: {user_vectors_top10[user_id]}")

# Save Step 8 & 10 results
print("\n[Saving Step 8 & 10 results...]")

# Save user vectors for Top-5
user_vectors_top5_df = pd.DataFrame(
    [{'user': user, **{f'PC{j+1}': vec[j] for j in range(5)}} 
     for user, vec in user_vectors_top5.items()]
)
user_vectors_top5_df.to_csv(os.path.join(RESULTS_DIR, 'step8_user_vectors_top5.csv'), index=False)

# Save user vectors for Top-10
user_vectors_top10_df = pd.DataFrame(
    [{'user': user, **{f'PC{j+1}': vec[j] for j in range(10)}} 
     for user, vec in user_vectors_top10.items()]
)
user_vectors_top10_df.to_csv(os.path.join(RESULTS_DIR, 'step10_user_vectors_top10.csv'), index=False)
print("Saved user projection vectors to results folder.")

# =============================================================================
# Step 9 & 11: User Similarity and Rating Prediction
# =============================================================================

print("\n" + "="*70)
print("Step 9 & 11: User Similarity and Rating Prediction")
print("="*70)

def cosine_similarity(vec_u, vec_v):
    """
    Calculate cosine similarity between two user vectors.
    
    Formula: Sim(u, v) = (UserVector_u · UserVector_v) / (||UserVector_u|| × ||UserVector_v||)
    
    Returns:
        float: Cosine similarity in range [-1, 1]
    """
    norm_u = np.linalg.norm(vec_u)
    norm_v = np.linalg.norm(vec_v)
    
    if norm_u == 0 or norm_v == 0:
        return 0.0
    
    return np.dot(vec_u, vec_v) / (norm_u * norm_v)

def predict_rating(target_user, target_item, user_vectors, item_means_dict, 
                   user_ratings, n_neighbors=20):
    """
    Predict rating for a target user on a target item.
    
    Formula: r_hat_{u,i} = μ_i + (Sum_{v in N} Sim(u,v) × (R_{v,i} - μ_i)) / (Sum_{v in N} |Sim(u,v)|)
    
    Args:
        target_user: The user for whom to predict
        target_item: The item to predict rating for
        user_vectors: Dictionary {user_id: user_vector}
        item_means_dict: Dictionary {item_id: mean}
        user_ratings: Dictionary {user_id: {item_id: rating}}
        n_neighbors: Number of nearest neighbors to use
    
    Returns:
        tuple: (predicted_rating, neighbors_used, neighbor_details)
    """
    target_vec = user_vectors[target_user]
    item_mean = item_means_dict.get(target_item, 0)
    
    # Calculate similarity with all other users who rated the target item
    similarities = []
    for user_id, user_vec in user_vectors.items():
        if user_id == target_user:
            continue
        # Check if this user rated the target item
        if user_id in user_ratings and target_item in user_ratings[user_id]:
            sim = cosine_similarity(target_vec, user_vec)
            actual_rating = user_ratings[user_id][target_item]
            similarities.append((user_id, sim, actual_rating))
    
    if not similarities:
        # No neighbors found, return item mean
        return item_mean, 0, []
    
    # Sort by similarity (descending) and take top N neighbors
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_neighbors = similarities[:n_neighbors]
    
    # Calculate weighted average
    numerator = 0.0
    denominator = 0.0
    neighbor_details = []
    
    for user_id, sim, actual_rating in top_neighbors:
        centered_rating = actual_rating - item_mean
        numerator += sim * centered_rating
        denominator += abs(sim)
        neighbor_details.append({
            'neighbor_user': user_id,
            'similarity': sim,
            'actual_rating': actual_rating,
            'centered_rating': centered_rating
        })
    
    if denominator == 0:
        return item_mean, len(top_neighbors), neighbor_details
    
    predicted = item_mean + (numerator / denominator)
    
    # Clip to valid rating range [1, 5]
    predicted = max(1.0, min(5.0, predicted))
    
    return predicted, len(top_neighbors), neighbor_details

# Make predictions for target users on target items
print("\n--- Rating Predictions for Target Users on Target Items ---")
n_neighbors = 20  # Number of nearest neighbors

prediction_results = []

for k_value, user_vectors, label in [(5, user_vectors_top5, "Top-5 PCs"), 
                                       (10, user_vectors_top10, "Top-10 PCs")]:
    print(f"\n=== Using {label} ===")
    
    for u_idx, target_user in enumerate(target_users, 1):
        print(f"\n--- Target User U{u_idx} (User {target_user}) ---")
        
        for i_idx, target_item in enumerate(target_items, 1):
            # Check if user already rated this item
            actual_rating = None
            if target_user in user_ratings and target_item in user_ratings[target_user]:
                actual_rating = user_ratings[target_user][target_item]
            
            predicted, n_used, neighbors = predict_rating(
                target_user, target_item, user_vectors, 
                item_means_dict, user_ratings, n_neighbors
            )
            
            item_mean = item_means_dict.get(target_item, 0)
            
            print(f"\n  Item I{i_idx} (Item {target_item}):")
            print(f"    Item Mean (μ_i): {item_mean:.4f}")
            print(f"    Neighbors Used: {n_used}")
            print(f"    Predicted Rating: {predicted:.4f}")
            if actual_rating is not None:
                print(f"    Actual Rating: {actual_rating}")
                print(f"    Error: {abs(predicted - actual_rating):.4f}")
            else:
                print(f"    Actual Rating: N/A (user hasn't rated this item)")
            
            # Show top 5 neighbors for detail
            if neighbors:
                print(f"    Top 5 Neighbors:")
                for n_info in neighbors[:5]:
                    print(f"      User {n_info['neighbor_user']}: "
                          f"Sim={n_info['similarity']:.4f}, "
                          f"Rating={n_info['actual_rating']}, "
                          f"Centered={n_info['centered_rating']:.4f}")
            
            prediction_results.append({
                'PCs': k_value,
                'PC_Label': label,
                'Target_User': f'U{u_idx}',
                'User_ID': target_user,
                'Target_Item': f'I{i_idx}',
                'Item_ID': target_item,
                'Item_Mean': item_mean,
                'Neighbors_Used': n_used,
                'Predicted_Rating': predicted,
                'Actual_Rating': actual_rating,
                'Error': abs(predicted - actual_rating) if actual_rating else None
            })

# Save prediction results
print("\n[Saving Step 9 & 11 results...]")
predictions_df = pd.DataFrame(prediction_results)
predictions_df.to_csv(os.path.join(RESULTS_DIR, 'step9_11_predictions.csv'), index=False)
print("Saved prediction results to results folder.")

# Summary table
print("\n" + "="*70)
print("PREDICTION SUMMARY")
print("="*70)

print("\n--- Rating Predictions Summary ---")
print(f"{'PCs':<10} {'User':<8} {'Item':<8} {'Predicted':<12} {'Actual':<10} {'Error':<10}")
print("-" * 58)
for result in prediction_results:
    actual_str = f"{result['Actual_Rating']}" if result['Actual_Rating'] else "N/A"
    error_str = f"{result['Error']:.4f}" if result['Error'] else "N/A"
    print(f"{result['PC_Label']:<10} {result['Target_User']:<8} {result['Target_Item']:<8} "
          f"{result['Predicted_Rating']:<12.4f} {actual_str:<10} {error_str:<10}")

print("\n" + "="*70)
print("[SUCCESS] All steps completed! (Steps 1-11)")
print("="*70)
