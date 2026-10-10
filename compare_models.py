
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ML5: Model Comparison and Selection

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# 1. Load ML2 and ML4 results
svd_results = pd.read_csv(REPORT_DIR / "svd_results.csv")
evaluation = pd.read_csv(REPORT_DIR / "evaluation_results.csv")

# 2. Extract popularity baseline
baseline = evaluation.loc[
    evaluation["model"] == "Popularity Baseline"
].iloc[0]

baseline_precision = baseline["precision_at_10"]

# 3. Prepare SVD evaluation data
svd_eval = evaluation[
    evaluation["model"].str.startswith("SVD k=")
].copy()

svd_eval["k"] = (
    svd_eval["model"].str.replace("SVD k=", "", regex=False)
    .astype(int)
)

# 4. Combine experiment and evaluation metrics
comparison = svd_eval.merge(
    svd_results[["k", "explained_variance_ratio"]],
    on="k",
    how="inner",
    validate="one_to_one"
)

assert len(comparison) == 4
assert baseline_precision > 0

# 5. Calculate relative improvement
comparison["precision_improvement_pct"] = (
    (comparison["precision_at_10"] / baseline_precision - 1) * 100
)

comparison = comparison.sort_values("k")

# 6. Select model using Precision@10
best = comparison.loc[comparison["precision_at_10"].idxmax()]

print("\nML5 Model Comparison:")
print(comparison.to_string(index=False))

print("\nSelected model:")
print(f"SVD k={int(best['k'])}")
print(f"Precision@10: {best['precision_at_10']:.4%}")
print(f"Recall@10: {best['recall_at_10']:.4%}")
print(f"Explained Variance: {best['explained_variance_ratio']:.4%}")
print(
    "Precision improvement over baseline: "
    f"{best['precision_improvement_pct']:.2f}%"
)

# 7. Save comparison report
comparison.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

# 8. Create comparison chart
fig, ax = plt.subplots(figsize=(8, 5))

labels = ["Popularity"] + [
    f"SVD k={k}" for k in comparison["k"]
]

precision_values = [
    baseline_precision * 100,
    *(comparison["precision_at_10"] * 100)
]

ax.bar(labels, precision_values)
ax.set_ylabel("Precision@10 (%)")
ax.set_title("Recommendation Precision Comparison")
ax.tick_params(axis="x", rotation=25)

fig.tight_layout()
fig.savefig(
    REPORT_DIR / "model_comparison.png",
    dpi=150
)
plt.close(fig)

# 9. Validation
assert np.isfinite(
    comparison["precision_improvement_pct"]
).all()

assert best["precision_at_10"] >= comparison["precision_at_10"].max()

print("\nReports saved.")
print("ML5 validation passed.")
