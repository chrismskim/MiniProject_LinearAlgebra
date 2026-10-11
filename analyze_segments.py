
from pathlib import Path
import pickle

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ML6: Customer Segment Analysis

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
K_VALUES = [3, 4, 5, 6]

# 1. Load training-derived customer embeddings
embeddings = joblib.load(
    DATA_DIR / "svd_models/customer_embeddings_k50.joblib"
)

train = pd.read_pickle(DATA_DIR / "train.pkl")

with open(DATA_DIR / "customer_map.pkl", "rb") as f:
    customer_map = pickle.load(f)

with open(DATA_DIR / "product_map.pkl", "rb") as f:
    product_map = pickle.load(f)

assert embeddings.shape == (len(customer_map), 50)

print("Customer embeddings:", embeddings.shape)

# 2. Compare cluster counts 3 through 6
silhouette_results = []

for n_clusters in K_VALUES:
    model = KMeans(
        n_clusters=n_clusters,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(embeddings)

    score = silhouette_score(
        embeddings,
        labels,
        sample_size=min(2000, len(embeddings)),
        random_state=RANDOM_STATE
    )

    silhouette_results.append({
        "n_clusters": n_clusters,
        "silhouette_score": score
    })

    print(f"k={n_clusters}: silhouette={score:.4f}")

silhouette_df = pd.DataFrame(silhouette_results)

best_k = int(
    silhouette_df.loc[
        silhouette_df["silhouette_score"].idxmax(),
        "n_clusters"
    ]
)

print("\nSelected clusters:", best_k)

# 3. Fit final clustering on 50D embeddings
model = KMeans(
    n_clusters=best_k,
    random_state=RANDOM_STATE,
    n_init=10
)

labels = model.fit_predict(embeddings)

# 4. Attach anonymous cluster labels to training data
train = train.copy()

train["customer_idx"] = train["Customer ID"].map(
    customer_map
)

train["StockCode"] = (
    train["StockCode"].astype(str).str.strip()
)

train = train.dropna(subset=["customer_idx"]).copy()
train["customer_idx"] = train["customer_idx"].astype(int)

train["cluster"] = labels[
    train["customer_idx"].to_numpy()
]

# 5. Find representative products
# Use number of distinct customers buying a product.
valid_products = set(product_map.keys())

product_data = train[
    train["StockCode"].isin(valid_products)
].copy()

product_counts = (
    product_data.groupby(
        ["cluster", "StockCode"]
    )["customer_idx"]
    .nunique()
    .reset_index(name="customer_count")
)

# Build representative product descriptions
descriptions = (
    product_data.dropna(subset=["Description"])
    .groupby("StockCode")["Description"]
    .agg(lambda x: x.astype(str).mode().iloc[0])
    .to_dict()
)

# 6. Find representative countries
# Count unique customers, not invoice rows.
customer_countries = (
    train.groupby("customer_idx")["Country"]
    .agg(lambda x: x.mode().iloc[0])
    .reset_index()
)

customer_countries["cluster"] = labels[
    customer_countries["customer_idx"].to_numpy()
]

country_counts = (
    customer_countries.groupby(
        ["cluster", "Country"]
    )["customer_idx"]
    .nunique()
    .reset_index(name="customer_count")
)

# 7. Create aggregate segment report
segment_rows = []

for cluster in range(best_k):
    n_customers = int(np.sum(labels == cluster))

    top_products = (
        product_counts[
            product_counts["cluster"] == cluster
        ]
        .sort_values(
            ["customer_count", "StockCode"],
            ascending=[False, True]
        )
        .head(5)
    )

    top_countries = (
        country_counts[
            country_counts["cluster"] == cluster
        ]
        .sort_values(
            ["customer_count", "Country"],
            ascending=[False, True]
        )
        .head(3)
    )

    product_names = [
        descriptions.get(code, str(code))
        for code in top_products["StockCode"]
    ]

    countries = top_countries["Country"].tolist()

    segment_rows.append({
        "segment": f"Segment {cluster}",
        "customer_count": n_customers,
        "customer_percentage": (
            n_customers / len(embeddings) * 100
        ),
        "top_products": " | ".join(product_names),
        "top_countries": " | ".join(countries)
    })

segment_df = pd.DataFrame(segment_rows)

assert segment_df["customer_count"].sum() == len(embeddings)

# 8. Project into 2D for visualization only
projection = PCA(
    n_components=2,
    random_state=RANDOM_STATE
).fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(9, 6))

scatter = ax.scatter(
    projection[:, 0],
    projection[:, 1],
    c=labels,
    cmap="tab10",
    s=8,
    alpha=0.6
)

ax.set_title("GiftLane Customer Segments")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")

fig.colorbar(scatter, ax=ax, label="Segment")
fig.tight_layout()

fig.savefig(
    REPORT_DIR / "customer_segments_final.png",
    dpi=150
)

plt.close(fig)

# 9. Save aggregate reports
silhouette_df.to_csv(
    REPORT_DIR / "silhouette_results_final.csv",
    index=False
)

segment_df.to_csv(
    REPORT_DIR / "segment_profiles.csv",
    index=False
)

print("\nSegment Profiles:")
print(segment_df.to_string(index=False))

print("\nML6 validation passed.")
