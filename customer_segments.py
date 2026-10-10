
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ML6: Customer Segmentation and Visualization

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")

REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
CLUSTER_COUNTS = [2, 3, 4, 5, 6]

# 1. Load customer embeddings from ML2
embeddings = joblib.load(
    DATA_DIR / "svd_models/customer_embeddings_k50.joblib"
)

print("Embedding shape:", embeddings.shape)

assert embeddings.shape == (5475, 50)
assert np.isfinite(embeddings).all()

# 2. Evaluate different cluster counts
results = []

for n_clusters in CLUSTER_COUNTS:
    print(f"\nEvaluating {n_clusters} clusters...")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(embeddings)

    # Sampled silhouette for efficient evaluation
    score = silhouette_score(
        embeddings,
        labels,
        sample_size=min(2000, len(embeddings)),
        random_state=RANDOM_STATE
    )

    results.append({
        "n_clusters": n_clusters,
        "silhouette_score": score
    })

    print(f"Silhouette score: {score:.4f}")

# 3. Select cluster count
results_df = pd.DataFrame(results)

best_row = results_df.loc[
    results_df["silhouette_score"].idxmax()
]

best_n_clusters = int(best_row["n_clusters"])

print("\nSelected number of clusters:", best_n_clusters)

# 4. Train final K-Means
final_model = KMeans(
    n_clusters=best_n_clusters,
    random_state=RANDOM_STATE,
    n_init=10
)

cluster_labels = final_model.fit_predict(embeddings)

# 5. Create aggregate cluster summary
cluster_sizes = np.bincount(
    cluster_labels,
    minlength=best_n_clusters
)

summary = pd.DataFrame({
    "cluster": np.arange(best_n_clusters),
    "customer_count": cluster_sizes,
    "customer_percentage": (
        cluster_sizes / len(embeddings) * 100
    )
})

assert summary["customer_count"].sum() == len(embeddings)

print("\nCluster Summary:")
print(summary.to_string(index=False))

# 6. Reduce embeddings to two dimensions for plotting
projection = PCA(
    n_components=2,
    random_state=RANDOM_STATE
).fit_transform(embeddings)

assert projection.shape == (5475, 2)

# 7. Visualize customer clusters
fig, ax = plt.subplots(figsize=(9, 6))

scatter = ax.scatter(
    projection[:, 0],
    projection[:, 1],
    c=cluster_labels,
    cmap="tab10",
    s=8,
    alpha=0.6
)

ax.set_title("Customer Segments in 2D SVD Embedding Space")
ax.set_xlabel("Projection Component 1")
ax.set_ylabel("Projection Component 2")

fig.colorbar(scatter, ax=ax, label="Cluster")
fig.tight_layout()

fig.savefig(
    REPORT_DIR / "customer_segments.png",
    dpi=150
)
plt.close(fig)

# 8. Save aggregate reports
results_df.to_csv(
    REPORT_DIR / "silhouette_results.csv",
    index=False
)

summary.to_csv(
    REPORT_DIR / "cluster_summary.csv",
    index=False
)

print("\nReports saved.")
print("ML6 validation passed.")
