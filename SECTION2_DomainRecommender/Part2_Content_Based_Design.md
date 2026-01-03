# Part 2: Content-Based Recommendation Strategy

## 1. Feature Extraction and Vector Space Model (3.1 & 3.2)

### **Text Feature Extraction: TF-IDF** (Selected)
*   **Approach:** Term Frequency-Inverse Document Frequency (TF-IDF) vectors.
*   **Preprocessing:** Tokenization, stop-word removal.
*   **Justification:**
    *   The `Text` field combines heterogeneous attributes: `Type` (e.g., "MOBA"), `Game` (e.g., "League of Legends"), and `Language` (e.g., "en").
    *   Terms like "en" (English) or popular game titles appear in a vast majority of documents. A simple Bag-of-Words model would over-emphasize these high-frequency terms, drowning out specific but crucial identifiers.
    *   **TF-IDF** naturally down-weights these globally common terms while boosting unique keywords that truly distinguish a streamer's content, resulting in more meaningful similarity scores.

### **Additional Features: Numerical Features** (Selected)
*   **Features:** `Rank`, `Avg Viewers`, `Followers`.
*   **Processing:** Normalized (Min-Max or Z-score) to match the scale of TF-IDF vectors.
*   **Justification:**
    *   **Addressing Long-Tail Bias:** The dataset statistics indicate a massive popularity skew (Top 10% receive 54.9% of ratings).
    *   Users exhibit a strong preference for "proven" content.
    *   Pure text similarity might recommend a niche streamer with 0 viewers who plays the same game as a popular one. By appending normalized popularity metrics, the model favors recommendations that align with the user's demonstrated preference for popular/high-quality streams.

## 2. User Profile Construction (4.1)

### **Method: Weighted Average** (Selected)
*   **Approach:** Weighted average of rated item features, where weights correspond to the rating values (e.g., Rating 1-5).
*   **Justification:**
    *   **Data Sparsity:** The median user has only **2 ratings**.
    *   **Rating Distribution:** 48.2% of ratings are '2' (Below Average). Only ~29% are \u2265 4.
    *   **Failure of Simple Average (\u2265 4):** Using only highly-rated items would result in **empty profiles** for a significant portion of users (those who only gave 2s or 3s), effectively treating active users as cold-start cases.
    *   **Benefit of Weighted:** This approach utilizes **100% of the available user interaction data**. Even low ratings provide critical signal; a rating of '1' or '2' pushes the user vector *away* from those content attributes, effectively modeling what the user *dislikes*.

## 3. Handling Cold-Start Users (4.2)

### **Strategy: Popular Item Features** (Selected)
*   **Approach:** Initialize new users' profiles using the centroid of the top k most popular items (by follower count or interaction frequency).
*   **Justification:**
    *   **Feasibility:** The provided "User Features" (`User ID`, `Profile Vector`) do not include demographic data (Age, Gender, Location), rendering the Demographic-based strategy impossible.
    *   **Statistical Alignment:** Given that the top 10% of streamers generate ~55% of all ratings, assume a new user is statistically most likely to be interested in these popular items. This is a safe, data-driven default for this specific distribution.

## 4. Similarity Computation (5.1 & 5.2)

*   **Metric:** Cosine Similarity between the User Profile Vector and all Item Vectors.
*   **Recommendation Generation:**
    1.  Compute cosine similarity scores for all unrated items.
    2.  Rank items by descending score.
    3.  Filter out items the user has already rated.
    4.  Return the Top-N (N=10, 20) items.

## 5. k-Nearest Neighbors (k-NN) Comparison (6)

*   **Approach:** Item-based k-NN.
*   **Comparison Point:** Compare the serendipity and coverage of the Content-Based approach (which may over-specialize) vs. k-NN (which relies on collaborative consumption patterns, likely reinforcing the popularity bias). The Content-Based approach is expected to better handle items in the "long tail" that have rich descriptions but few ratings, whereas k-NN will struggle with these items.
