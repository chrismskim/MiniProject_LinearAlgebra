
from pathlib import Path
import time

import scipy.sparse as sp
from sklearn.decomposition import PCA

# Load the same training matrix used for SVD
A = sp.load_npz(
    Path("data/processed/customer_product_matrix.npz")
).tocsr()

print("Matrix shape:", A.shape)

# Fit PCA with the same number of components
print("Training PCA with k=50...", flush=True)

start = time.perf_counter()

model = PCA(
    n_components=50,
    svd_solver="arpack",
    random_state=42
)

embeddings = model.fit_transform(A)

elapsed = time.perf_counter() - start

print("PCA embedding shape:", embeddings.shape)
print(
    "Explained variance:",
    f"{model.explained_variance_ratio_.sum():.4%}"
)
print(f"PCA fit time: {elapsed:.4f} seconds")

assert embeddings.shape == (5475, 50)

print("PCA validation passed.")
