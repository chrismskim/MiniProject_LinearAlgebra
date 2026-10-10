
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

# ML4: Recommendation Evaluation

DATA_DIR = Path("data/processed")
RECOMMENDATION_DIR = DATA_DIR / "recommendations"
REPORT_DIR = Path("reports")

TOP_N = 10
K_VALUES = [10, 20, 50, 100]

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load held-out test data
test = pd.read_pickle(DATA_DIR / "test.pkl")

with open(DATA_DIR / "customer_map.pkl", "rb") as f:
    customer_map = pickle.load(f)

with open(DATA_DIR / "product_map.pkl", "rb") as f:
    product_map = pickle.load(f)

print("Original held-out test rows:", len(test))

# 2. Normalize product codes to match DE4
test = test.copy()
test["StockCode"] = test["StockCode"].astype(str).str.strip()

# 3. Map customers and products to training indices
test["customer_idx"] = test["Customer ID"].map(customer_map)
test["product_idx"] = test["StockCode"].map(product_map)

cold_start_customers = test["customer_idx"].isna().sum()
unavailable_products = test["product_idx"].isna().sum()

print("Rows with unknown customers:", cold_start_customers)
print("Rows with unavailable products:", unavailable_products)

# 4. Keep only evaluable purchases
test = test.dropna(
    subset=["customer_idx", "product_idx"]
).copy()

test["customer_idx"] = test["customer_idx"].astype(int)
test["product_idx"] = test["product_idx"].astype(int)

# Remove duplicate customer-product pairs
test = test.drop_duplicates(
    subset=["customer_idx", "product_idx"]
)

# 5. Build ground truth
ground_truth = (
    test.groupby("customer_idx")["product_idx"]
    .apply(set)
    .to_dict()
)

eligible_customers = sorted(ground_truth.keys())

print("Unique evaluable customer-product pairs:", len(test))
print("Eligible evaluation customers:", len(eligible_customers))

assert len(eligible_customers) > 0

# 6. Define evaluation function
def evaluate_model(recommendations, model_name):
    precision_scores = []
    recall_scores = []
    hit_count = 0

    for customer_idx in eligible_customers:
        actual_products = ground_truth[customer_idx]
        predicted_products = set(
            recommendations[customer_idx].tolist()
        )

        hits = len(actual_products.intersection(predicted_products))

        precision_scores.append(hits / TOP_N)
        recall_scores.append(hits / len(actual_products))

        if hits > 0:
            hit_count += 1

    return {
        "model": model_name,
        "eligible_customers": len(eligible_customers),
        "precision_at_10": np.mean(precision_scores),
        "recall_at_10": np.mean(recall_scores),
        "hit_rate_at_10": hit_count / len(eligible_customers)
    }


# 7. Load and evaluate recommendation outputs
models = {
    "Popularity Baseline": "popularity_top10.npy",
    "SVD k=10": "svd_top10_k10.npy",
    "SVD k=20": "svd_top10_k20.npy",
    "SVD k=50": "svd_top10_k50.npy",
    "SVD k=100": "svd_top10_k100.npy"
}

results = []

for model_name, filename in models.items():
    recommendations = np.load(RECOMMENDATION_DIR / filename)

    assert recommendations.shape[1] == TOP_N
    assert max(eligible_customers) < recommendations.shape[0]

    result = evaluate_model(recommendations, model_name)
    results.append(result)

    print(
        f"{model_name}: "
        f"Precision@10={result['precision_at_10']:.4f}, "
        f"Recall@10={result['recall_at_10']:.4f}, "
        f"HitRate@10={result['hit_rate_at_10']:.4f}"
    )

# 8. Save aggregate results
report = pd.DataFrame(results)

report.to_csv(
    REPORT_DIR / "evaluation_results.csv",
    index=False
)

# 9. Validate metrics
for metric in [
    "precision_at_10",
    "recall_at_10",
    "hit_rate_at_10"
]:
    assert report[metric].between(0, 1).all()

print("\nML4 Evaluation Results:")
print(report.to_string(index=False))
print("\nReport saved to reports/evaluation_results.csv")
print("ML4 validation passed.")
