
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

# ML7: Purchase Value Sensitivity Analysis

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

K = 50
TOP_N = 10
RANDOM_STATE = 42

# 1. Load training and held-out data
train = pd.read_pickle(DATA_DIR / "train.pkl")
test = pd.read_pickle(DATA_DIR / "test.pkl")

with open(DATA_DIR / "customer_map.pkl", "rb") as f:
    customer_map = pickle.load(f)

with open(DATA_DIR / "product_map.pkl", "rb") as f:
    product_map = pickle.load(f)

n_customers = len(customer_map)
n_products = len(product_map)

# 2. Normalize product codes
train = train.copy()
test = test.copy()

train["StockCode"] = train["StockCode"].astype(str).str.strip()
test["StockCode"] = test["StockCode"].astype(str).str.strip()

# 3. Map training purchases
train["customer_idx"] = train["Customer ID"].map(customer_map)
train["product_idx"] = train["StockCode"].map(product_map)

train = train.dropna(
    subset=["customer_idx", "product_idx"]
).copy()

train["customer_idx"] = train["customer_idx"].astype(int)
train["product_idx"] = train["product_idx"].astype(int)

# Aggregate quantity before applying transformations
purchases = (
    train.groupby(["customer_idx", "product_idx"])["Quantity"]
    .sum()
    .reset_index()
)

# 4. Map held-out test purchases
test["customer_idx"] = test["Customer ID"].map(customer_map)
test["product_idx"] = test["StockCode"].map(product_map)

test = test.dropna(
    subset=["customer_idx", "product_idx"]
).copy()

test["customer_idx"] = test["customer_idx"].astype(int)
test["product_idx"] = test["product_idx"].astype(int)

test = test.drop_duplicates(
    subset=["customer_idx", "product_idx"]
)

ground_truth = (
    test.groupby("customer_idx")["product_idx"]
    .apply(set)
    .to_dict()
)

eligible_customers = sorted(ground_truth)

assert len(eligible_customers) > 0

# 5. Build matrix with selected purchase values
def build_matrix(method):
    quantities = purchases["Quantity"].to_numpy()

    if method == "Count":
        values = quantities.astype(float)
    elif method == "Binary":
        values = np.ones(len(quantities), dtype=float)
    elif method == "Log":
        values = np.log1p(quantities)
    else:
        raise ValueError(f"Unknown method: {method}")

    matrix = sp.csr_matrix(
        (
            values,
            (
                purchases["customer_idx"].to_numpy(),
                purchases["product_idx"].to_numpy()
            )
        ),
        shape=(n_customers, n_products)
    )

    assert matrix.nnz == len(purchases)
    return matrix


# 6. Evaluate the same recommendation protocol as ML4
def evaluate(matrix):
    model = TruncatedSVD(
        n_components=K,
        algorithm="randomized",
        n_iter=7,
        random_state=RANDOM_STATE
    )

    embeddings = model.fit_transform(matrix)

    hits_by_customer = []
    precision_scores = []
    recall_scores = []

    for customer_idx in eligible_customers:
        scores = (
            embeddings[customer_idx] @ model.components_
        )

        scores = scores.copy()

        start = matrix.indptr[customer_idx]
        end = matrix.indptr[customer_idx + 1]
        purchased = matrix.indices[start:end]

        scores[purchased] = -np.inf

        top10 = np.argsort(
            -scores, kind="stable"
        )[:TOP_N]

        actual = ground_truth[customer_idx]
        hits = len(set(top10).intersection(actual))

        hits_by_customer.append(int(hits > 0))
        precision_scores.append(hits / TOP_N)
        recall_scores.append(hits / len(actual))

    return {
        "hit_rate_at_10": np.mean(hits_by_customer),
        "precision_at_10": np.mean(precision_scores),
        "recall_at_10": np.mean(recall_scores),
        "explained_variance_ratio": (
            model.explained_variance_ratio_.sum()
        )
    }


# 7. Run sensitivity experiments
results = []

for method in ["Count", "Binary", "Log"]:
    print(f"\nEvaluating purchase value: {method}", flush=True)

    matrix = build_matrix(method)
    metrics = evaluate(matrix)

    result = {
        "purchase_value": method,
        "k": K,
        "eligible_customers": len(eligible_customers),
        "matrix_shape": str(matrix.shape),
        **metrics
    }

    results.append(result)

    print(f"Hit Rate@10: {metrics['hit_rate_at_10']:.4%}")
    print(f"Precision@10: {metrics['precision_at_10']:.4%}")
    print(f"Recall@10: {metrics['recall_at_10']:.4%}")

# 8. Save results
report = pd.DataFrame(results)

assert report["eligible_customers"].nunique() == 1

for metric in [
    "hit_rate_at_10",
    "precision_at_10",
    "recall_at_10",
    "explained_variance_ratio"
]:
    assert report[metric].between(0, 1).all()

report.to_csv(
    REPORT_DIR / "sensitivity_results.csv",
    index=False
)

print("\nML7 Sensitivity Results:")
print(report.to_string(index=False))
print("\nML7 validation passed.")
