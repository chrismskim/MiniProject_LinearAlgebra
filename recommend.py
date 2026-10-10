
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ML3: Product Recommendation System

MATRIX_PATH = Path("data/processed/customer_product_matrix.npz")
MODEL_DIR = Path("data/processed/svd_models")
OUTPUT_DIR = Path("data/processed/recommendations")

K_VALUES = [10, 20, 50, 100]
TOP_N = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load training-only customer-product matrix
A = sp.load_npz(MATRIX_PATH).tocsr()
n_customers, n_products = A.shape

print("Matrix shape:", A.shape)

# 2. Calculate popularity baseline from training purchases
# A stores log1p(quantity), so recover original quantity.
product_popularity = np.asarray(
    np.expm1(A).sum(axis=0)
).ravel()

popularity_ranking = np.argsort(
    -product_popularity,
    kind="stable"
)


def top_n_unpurchased(scores, purchased_indices, n=TOP_N):
    """Return the highest-scoring products not purchased in training."""
    eligible_scores = scores.copy()
    eligible_scores[purchased_indices] = -np.inf

    eligible_count = np.isfinite(eligible_scores).sum()
    if eligible_count < n:
        raise ValueError("Not enough eligible products.")

    # Stable sorting for reproducible tie-breaking
    return np.argsort(-eligible_scores, kind="stable")[:n]


def popularity_recommendations(customer_index):
    """Recommend popular products not purchased by this customer."""
    start = A.indptr[customer_index]
    end = A.indptr[customer_index + 1]
    purchased = set(A.indices[start:end])

    recommended = [
        product_index
        for product_index in popularity_ranking
        if product_index not in purchased
    ]

    return np.asarray(recommended[:TOP_N], dtype=int)


def svd_recommendations(customer_index, model, embeddings):
    """Generate SVD recommendations for one customer."""
    # Customer embedding: shape (k,)
    customer_vector = embeddings[customer_index]

    # components_: shape (k, n_products)
    # Result: predicted score for every product
    scores = customer_vector @ model.components_

    start = A.indptr[customer_index]
    end = A.indptr[customer_index + 1]
    purchased_indices = A.indices[start:end]

    return top_n_unpurchased(scores, purchased_indices)


def validate_recommendations(customer_index, recommendations):
    """Verify recommendation length, uniqueness, and novelty."""
    start = A.indptr[customer_index]
    end = A.indptr[customer_index + 1]
    purchased = set(A.indices[start:end])

    assert len(recommendations) == TOP_N
    assert len(set(recommendations)) == TOP_N
    assert not purchased.intersection(recommendations)
    assert all(0 <= idx < n_products for idx in recommendations)


# 3. Build popularity baseline
print("\nGenerating popularity baseline...")

baseline_results = []

for customer_idx in range(n_customers):
    recommended = popularity_recommendations(customer_idx)
    validate_recommendations(customer_idx, recommended)

    baseline_results.append(recommended)

baseline_results = np.asarray(baseline_results, dtype=np.int32)

np.save(
    OUTPUT_DIR / "popularity_top10.npy",
    baseline_results
)

print("Popularity recommendation shape:", baseline_results.shape)
print("Popularity baseline validation passed.")


# 4. Build recommendations for each SVD model
for k in K_VALUES:
    print(f"\nGenerating SVD recommendations for k={k}...")

    model = joblib.load(MODEL_DIR / f"svd_k{k}.joblib")
    embeddings = joblib.load(
        MODEL_DIR / f"customer_embeddings_k{k}.joblib"
    )

    assert embeddings.shape == (n_customers, k)
    assert model.components_.shape == (k, n_products)

    results = []

    for customer_idx in range(n_customers):
        recommended = svd_recommendations(
            customer_idx,
            model,
            embeddings
        )

        validate_recommendations(customer_idx, recommended)
        results.append(recommended)

    results = np.asarray(results, dtype=np.int32)

    np.save(
        OUTPUT_DIR / f"svd_top10_k{k}.npy",
        results
    )

    print("Recommendation shape:", results.shape)
    print(f"SVD k={k} validation passed.")

print("\nML3 recommendation generation completed.")
print("Outputs saved to:", OUTPUT_DIR)
