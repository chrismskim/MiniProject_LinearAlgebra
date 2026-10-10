# Model Decisions

## ML1: Dimensionality Reduction Method Selection

### Problem

GiftLane aims to recommend relevant products to customers based on historical purchasing behavior.

The training dataset was transformed into a sparse customer-product matrix with the following characteristics:

| Metric | Value |
|---|---:|
| Customers | 5,475 |
| Products | 3,689 |
| Non-zero Entries | 413,983 |
| Sparsity | 97.95% |

The objective is to reduce the dimensionality of the customer-product matrix while preserving useful patterns in customer purchasing behavior.

### Candidate Methods

**Principal Component Analysis (PCA)**

PCA identifies directions of maximum variance in mean-centered data. However, centering a sparse matrix can make it dense, increasing memory requirements.

**Truncated Singular Value Decomposition (Truncated SVD)**

Truncated SVD approximates the original matrix using a limited number of singular components. It can operate directly on sparse matrices without requiring mean-centering.

### Selected Method: Truncated SVD

Truncated SVD is selected as the primary dimensionality reduction method for the following reasons:

1. The customer-product matrix is highly sparse (97.95%).
2. Truncated SVD supports sparse matrices without converting the entire matrix to dense format.
3. It provides lower-dimensional customer representations that can capture shared purchasing patterns.
4. The resulting representations can be used to generate product recommendations.

### Mathematical Representation

For a customer-product matrix A, singular value decomposition is expressed as:

A = U Σ Vᵀ

A rank-k approximation is:

Aₖ = Uₖ Σₖ Vₖᵀ

Here, k represents the number of retained singular components.

The goal is to represent customers and products using fewer dimensions while preserving useful purchasing information.

### Evaluation Plan

Truncated SVD will be tested using multiple values of k.

Recommendation quality will be evaluated against held-out customer-product purchases.

A popularity-based top-10 recommendation baseline will be used for comparison.

Only training data will be used when fitting the model to prevent test-data leakage.

### Conclusion

Truncated SVD is the primary model because it is computationally suitable for large, sparse purchase matrices.

Its recommendation quality will be assessed experimentally rather than assumed.