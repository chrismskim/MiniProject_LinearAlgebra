
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

# ML2: Truncated SVD Experiments

MATRIX_PATH = Path("data/processed/customer_product_matrix.npz")
MODEL_DIR = Path("data/processed/svd_models")
REPORT_DIR = Path("reports")

K_VALUES = [10, 20, 50, 100, 200]
RANDOM_STATE = 42

# 1. Load the training customer-product matrix
A = sp.load_npz(MATRIX_PATH)

# Calculate squared Frobenius norm of the original matrix
matrix_norm_squared = float(A.multiply(A).sum())

assert matrix_norm_squared > 0

print("Original matrix shape:", A.shape)
print("Non-zero entries:", A.nnz)

assert sp.isspmatrix_csr(A)
assert max(K_VALUES) < min(A.shape)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

results = []

# 2. Train SVD models
for k in K_VALUES:
    print(f"\nTraining Truncated SVD with k={k}...")

    model = TruncatedSVD(
        n_components=k,
        algorithm="randomized",
        n_iter=7,
        random_state=RANDOM_STATE
    )

    customer_embeddings = model.fit_transform(A)

    # 3. Explained variance
    explained_variance = model.explained_variance_ratio_.sum()
    # Calculate relative reconstruction error
    components = model.components_

    projected = A @ components.T

    cross_term = np.sum(customer_embeddings * projected)

    reconstruction_norm_squared = np.sum(
        (customer_embeddings.T @ customer_embeddings)
        * (components @ components.T)
        )

    error_squared = (
        matrix_norm_squared
        - 2 * cross_term
        + reconstruction_norm_squared
    )

    relative_error = np.sqrt(
        max(float(error_squared), 0.0) / matrix_norm_squared
    )

    print(f"Relative reconstruction error: {relative_error:.4%}")

    print("Embedding shape:", customer_embeddings.shape)
    print(f"Explained variance: {explained_variance:.4%}")

    # 4. Validation
    assert customer_embeddings.shape == (A.shape[0], k)
    assert np.isfinite(customer_embeddings).all()
    assert 0 <= explained_variance <= 1 + 1e-10

    # 5. Save fitted model and embeddings
    joblib.dump(
        model,
        MODEL_DIR / f"svd_k{k}.joblib"
    )

    joblib.dump(
        customer_embeddings,
        MODEL_DIR / f"customer_embeddings_k{k}.joblib"
    )

    results.append({
        "k": k,
        "customers": A.shape[0],
        "original_features": A.shape[1],
        "reduced_features": k,
        "explained_variance_ratio": explained_variance,
        "relative_reconstruction_error": relative_error
        })

# 6. Save experiment results
report = pd.DataFrame(results)

report.to_csv(
    REPORT_DIR / "svd_results.csv",
    index=False
)

# Plot reconstruction error
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    report["k"],
    report["relative_reconstruction_error"],
    marker="o"
)

ax.set_xlabel("Number of Components (k)")
ax.set_ylabel("Relative Reconstruction Error")
ax.set_title("Truncated SVD Reconstruction Error vs k")
ax.grid(True, alpha=0.3)

fig.tight_layout()

fig.savefig(
    REPORT_DIR / "svd_reconstruction_error.png",
    dpi=150
)

plt.close(fig)

print("Reconstruction error plot saved.")

print("\nML2 Results:")
print(report.to_string(index=False))
print("\nReport saved to reports/svd_results.csv")
print("ML2 validation passed.")
