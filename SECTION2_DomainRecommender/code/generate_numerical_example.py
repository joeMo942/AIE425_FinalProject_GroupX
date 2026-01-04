
"""
Assignment Part 7: Complete Numerical Example
==============================================
Generates a step-by-step numerical example for the report.
Shows:
1. TF-IDF vectors for 3 sample items
2. User Profile construction (weighted average)
3. Similarity calculation
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Directories
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

def main():
    print("="*60)
    print("GENERATING NUMERICAL EXAMPLE (PART 7)")
    print("="*60)
    
    # Load Content-Based Model
    model_path = RESULTS_DIR / "content_based_model.pkl"
    if not model_path.exists():
        print("[ERROR] Model not found. Run content_based.py first.")
        return

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    tfidf_vectorizer = model['tfidf']
    item_to_idx = model['item_to_idx']
    idx_to_item = model['idx_to_item']
    features = model['item_features']
    
    # feature names
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Select 3 sample items
    # Try getting distinct genre items if possible, else random
    sample_streamers = ['ninja', 'shroud', 'chess'] # Popular distinct ones
    # verify they exist
    samples = [s for s in sample_streamers if s in item_to_idx]
    if len(samples) < 3:
        # Fallback to first 3
        samples = list(idx_to_item.values())[:3]
        
    output = []
    output.append("PART 7: COMPLETE NUMERICAL EXAMPLE\n")
    output.append("="*40 + "\n\n")
    
    # 1. TF-IDF Vectors
    output.append("1. Feature Extraction (TF-IDF)\n")
    output.append("-" * 30 + "\n")
    
    for streamer in samples:
        idx = item_to_idx[streamer]
        vec = features[idx]
        
        # Get top 5 features
        # Note: 'features' is dense combined (0-500 are TF-IDF)
        tfidf_part = vec[:500] 
        top_indices = np.argsort(tfidf_part)[::-1][:5]
        
        output.append(f"Item: {streamer}")
        output.append("  Top TF-IDF Terms:")
        for feat_idx in top_indices:
            if tfidf_part[feat_idx] > 0:
                term = feature_names[feat_idx]
                score = tfidf_part[feat_idx]
                output.append(f"    - '{term}': {score:.4f}")
        output.append("\n")

    # 2. User Profile Construction
    output.append("2. User Profile Construction\n")
    output.append("-" * 30 + "\n")
    
    # Simulate a user rating 2 items
    user_history = [
        (samples[0], 5.0), # Loved Item A
        (samples[1], 3.0)  # Liked Item B
    ]
    
    output.append("Hypothetical User History:")
    for s, r in user_history:
        output.append(f"  - Rated '{s}': {r} stars")
        
    # Calculate profile vector manually for display
    idx_a = item_to_idx[samples[0]]
    idx_b = item_to_idx[samples[1]]
    vec_a = features[idx_a]
    vec_b = features[idx_b]
    
    # Weighted avg
    total_weight = 5.0 + 3.0
    w_a = 5.0 / total_weight
    w_b = 3.0 / total_weight
    
    user_profile = (vec_a * 5.0 + vec_b * 3.0) / total_weight
    
    output.append("\nProfile Calculation (Weighted Average):")
    output.append(f"  Weight A = 5.0 / 8.0 = {w_a:.3f}")
    output.append(f"  Weight B = 3.0 / 8.0 = {w_b:.3f}")
    output.append("  Profile_Vec = (Vec_A * 0.625) + (Vec_B * 0.375)")
    
    # Show user profile top terms
    output.append("\nUser Profile Top Terms:")
    tfidf_part = user_profile[:500]
    top_indices = np.argsort(tfidf_part)[::-1][:5]
    for feat_idx in top_indices:
            if tfidf_part[feat_idx] > 0:
                term = feature_names[feat_idx]
                score = tfidf_part[feat_idx]
                output.append(f"    - '{term}': {score:.4f}")
    
    # 3. Similarity & Recommendation
    output.append("\n3. Similarity & Recommendation\n")
    output.append("-" * 30 + "\n")
    
    target_item = samples[2]
    idx_target = item_to_idx[target_item]
    vec_target = features[idx_target]
    
    # Cosine Sim
    # sim = dot(u, v) / (norm(u)*norm(v))
    # Sklearn does this efficiently
    sim = cosine_similarity(user_profile.reshape(1,-1), vec_target.reshape(1,-1))[0][0]
    
    output.append(f"Target Item: {target_item}")
    output.append(f"Cosine Similarity(User, {target_item}): {sim:.4f}")
    
    if sim > 0.1:
        output.append("-> RECOMMEND (High Similarity)")
    else:
        output.append("-> IGNORE (Low Similarity)")

    # Save to file
    out_file = RESULTS_DIR / "Part7_Numerical_Example.txt"
    with open(out_file, 'w') as f:
        f.writelines([line + "\n" if not line.endswith("\n") else line for line in output])
        
    print(f"\n[DONE] Generated report: {out_file}")
    print("Review contents for your report.")

if __name__ == "__main__":
    main()
