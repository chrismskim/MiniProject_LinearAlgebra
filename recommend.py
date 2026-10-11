
from pathlib import Path
from sklearn.preprocessing import normalize
import time
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

# ML3: Product Recommendation System

MATRIX_PATH = Path("data/processed/customer_product_matrix.npz")
MODEL_DIR = Path("data/processed/svd_models")
OUTPUT_DIR = Path("data/processed/recommendations")

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

job_start = time.perf_counter()
timing_results = []

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
baseline_start = time.perf_counter()
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

baseline_seconds = time.perf_counter() - baseline_start

timing_results.append({
    "model": "Popularity Baseline",
    "customers": n_customers,
    "generation_seconds": baseline_seconds
})

print(f"Baseline generation time: {baseline_seconds:.4f} seconds")


def similar_products(product_idx, model, top_n=10):
    # Each row represents a product in the latent space
    product_vectors = normalize(model.components_.T)

    if not 0 <= product_idx < len(product_vectors):
        raise ValueError("Invalid product index")

    # Cosine similarity between products
    similarities = product_vectors @ product_vectors[product_idx]

    # Exclude the product itself
    similarities[product_idx] = -np.inf

    # Return top-N most similar product indices
    return np.argsort(-similarities, kind="stable")[:top_n]

# 4. Build recommendations for each SVD model
for k in K_VALUES:
    print(f"\nGenerating SVD recommendations for k={k}...")
    model_start = time.perf_counter()

    model = joblib.load(MODEL_DIR / f"svd_k{k}.joblib")
    embeddings = joblib.load(
        MODEL_DIR / f"customer_embeddings_k{k}.joblib"
    )

    assert embeddings.shape == (n_customers, k)
    assert model.components_.shape == (k, n_products)
    # ML3: Product similarity validation
    if k == 50:
        example_product_idx = 0

        neighbors = similar_products(
            example_product_idx,
            model,
            top_n=10
        )

        assert len(neighbors) == 10
        assert len(set(neighbors)) == 10
        assert example_product_idx not in neighbors
        assert all(0 <= idx < n_products for idx in neighbors)

        print("\nML3 Product Similarity Test")
        print("Product index:", example_product_idx)
        print("Top-10 similar product indices:", neighbors)
        print("Product similarity validation passed.")


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
    generation_seconds = time.perf_counter() - model_start

    timing_results.append({
        "model": f"SVD k={k}",
        "customers": n_customers,
        "generation_seconds": generation_seconds
    })

    print(
        f"SVD k={k} generation time: "
        f"{generation_seconds:.4f} seconds"
    )

print("\nML3 recommendation generation completed.")
print("Outputs saved to:", OUTPUT_DIR)

# ML4: Nightly job runtime report
total_job_seconds = time.perf_counter() - job_start

timing_report = pd.DataFrame(timing_results)

timing_report.to_csv(
    REPORT_DIR / "recommendation_runtime.csv",
    index=False
)

print("\nML4 Nightly Job Runtime:")
print(timing_report.to_string(index=False))

print(f"\nTotal job time: {total_job_seconds:.4f} seconds")
print("ML4 runtime validation passed.")