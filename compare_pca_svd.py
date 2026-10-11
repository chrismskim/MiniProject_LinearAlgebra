
from pathlib import Path
import json
import pickle
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA, TruncatedSVD


DATA_DIR = Path("data/processed")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

K = 50
TOP_N = 10
RANDOM_STATE = 42


def load_data():
    A = sp.load_npz(
        DATA_DIR / "customer_product_matrix.npz"
    ).tocsr()

    test = pd.read_pickle(DATA_DIR / "test.pkl")

    with open(DATA_DIR / "customer_map.pkl", "rb") as f:
        customer_map = pickle.load(f)

    with open(DATA_DIR / "product_map.pkl", "rb") as f:
        product_map = pickle.load(f)

    return A, test, customer_map, product_map


def prepare_ground_truth(test, customer_map, product_map):
    test = test.copy()
    test["StockCode"] = test["StockCode"].astype(str).str.strip()

    test["customer_idx"] = test["Customer ID"].map(customer_map)
    test["product_idx"] = test["StockCode"].map(product_map)

    test = test.dropna(
        subset=["customer_idx", "product_idx"]
    )

    test["customer_idx"] = test["customer_idx"].astype(int)
    test["product_idx"] = test["product_idx"].astype(int)

    test = test.drop_duplicates(
        subset=["customer_idx", "product_idx"]
    )

    return (
        test.groupby("customer_idx")["product_idx"]
        .apply(set)
        .to_dict()
    )


def evaluate_hit_rate(A, model, embeddings, ground_truth):
    hits = []

    for customer_idx, actual_products in ground_truth.items():
        scores = embeddings[customer_idx] @ model.components_

        # PCA scores are centered, so add training column means.
        if isinstance(model, PCA):
            scores = scores + model.mean_

        scores = scores.copy()

        start = A.indptr[customer_idx]
        end = A.indptr[customer_idx + 1]
        scores[A.indices[start:end]] = -np.inf

        top10 = np.argsort(
            -scores, kind="stable"
        )[:TOP_N]

        hits.append(
            int(bool(set(top10).intersection(actual_products)))
        )

    return float(np.mean(hits))


def get_peak_memory_mb():
    peak = resource.getrusage(
        resource.RUSAGE_SELF
    ).ru_maxrss

    # macOS reports bytes; Linux generally reports KiB.
    if sys.platform == "darwin":
        return peak / (1024 ** 2)

    return peak / 1024


def run_experiment(method):
    A, test, customer_map, product_map = load_data()

    ground_truth = prepare_ground_truth(
        test, customer_map, product_map
    )

    if method == "PCA":
        model = PCA(
            n_components=K,
            svd_solver="arpack",
            random_state=RANDOM_STATE
        )
    else:
        model = TruncatedSVD(
            n_components=K,
            algorithm="randomized",
            n_iter=7,
            random_state=RANDOM_STATE
        )

    start = time.perf_counter()
    embeddings = model.fit_transform(A)
    fit_seconds = time.perf_counter() - start

    hit_rate = evaluate_hit_rate(
        A, model, embeddings, ground_truth
    )

    # Store the factors needed to reconstruct predictions.
    factor_bytes = embeddings.nbytes + model.components_.nbytes

    if method == "PCA":
        factor_bytes += model.mean_.nbytes

    # Top product codes contributing to component 1
    reverse_product_map = {
        idx: code for code, idx in product_map.items()
    }

    component_1 = model.components_[0]
    top_indices = np.argsort(
        -np.abs(component_1)
    )[:5]

    top_product_codes = [
        str(reverse_product_map[int(idx)])
        for idx in top_indices
    ]

    result = {
        "method": method,
        "k": K,
        "fit_seconds": fit_seconds,
        "peak_process_rss_mb": get_peak_memory_mb(),
        "stored_factors_mb": factor_bytes / (1024 ** 2),
        "explained_variance_ratio": float(
            model.explained_variance_ratio_.sum()
        ),
        "hit_rate_at_10": hit_rate,
        "eligible_customers": len(ground_truth),
        "component_1_top_products": ", ".join(
            top_product_codes
        )
    }

    return result


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--model":
        result = run_experiment(sys.argv[2])
        print(json.dumps(result))
        return

    results = []

    for method in ["PCA", "SVD"]:
        print(f"Evaluating {method}...", flush=True)

        # Independent processes make peak memory measurements
        # more comparable.
        completed = subprocess.run(
            [sys.executable, __file__, "--model", method],
            check=True,
            capture_output=True,
            text=True
        )

        results.append(json.loads(completed.stdout))

    report = pd.DataFrame(results)

    assert report["eligible_customers"].nunique() == 1
    assert report["k"].nunique() == 1
    assert report["hit_rate_at_10"].between(0, 1).all()

    baseline = pd.read_csv(
        REPORT_DIR / "evaluation_results.csv"
    )

    baseline_hit_rate = baseline.loc[
        baseline["model"] == "Popularity Baseline",
        "hit_rate_at_10"
    ].iloc[0]

    report["baseline_hit_rate_at_10"] = baseline_hit_rate

    report.to_csv(
        REPORT_DIR / "pca_vs_svd.csv",
        index=False
    )

    print("\nML5 PCA vs SVD Comparison:")
    print(report.to_string(index=False))
    print("\nReport saved to reports/pca_vs_svd.csv")
    print("ML5 comparison validation passed.")


if __name__ == "__main__":
    main()
