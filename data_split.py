
import pandas as pd
from pathlib import Path

# ==========================================
# DE3: Time-based Train/Test Split
# ==========================================

# 1. Load cleaned dataset
df = pd.read_pickle("data/processed/cleaned_retail.pkl")

print("Original cleaned rows:", len(df))
print("Earliest date:", df["InvoiceDate"].min())
print("Latest date:", df["InvoiceDate"].max())

# 2. Define the last two calendar months as the test period
latest_date = df["InvoiceDate"].max()
test_start = latest_date - pd.DateOffset(months=2)

print("\nTest period starts:", test_start)

# 3. Split data by time
train = df[df["InvoiceDate"] < test_start].copy()
test_period = df[df["InvoiceDate"] >= test_start].copy()

print("\nTraining rows:", len(train))
print("Test-period rows:", len(test_period))

# 4. Identify products already purchased by each customer
previous_purchases = (
    train[["Customer ID", "StockCode"]]
    .drop_duplicates()
    .assign(purchased_before=True)
)

# 5. Match test purchases against training history
test_candidates = test_period.merge(
    previous_purchases,
    on=["Customer ID", "StockCode"],
    how="left",
    validate="many_to_one"
)

# 6. Keep only purchases of previously unseen products
test = test_candidates[
    test_candidates["purchased_before"].isna()
].drop(columns=["purchased_before"]).copy()

# 7. Keep customers that exist in the training data
train_customers = set(train["Customer ID"])

test = test[
    test["Customer ID"].isin(train_customers)
].copy()

print("\nHeld-out new-product rows:", len(test))
print("Training customers:", train["Customer ID"].nunique())
print("Eligible test customers:", test["Customer ID"].nunique())

# 8. Validate chronological separation
assert train["InvoiceDate"].max() < test_start

assert (test["InvoiceDate"] >= test_start).all()

assert len(train) + len(test_period) == len(df)

# 9. Validate unseen customer-product pairs
train_pairs = pd.MultiIndex.from_frame(
    train[["Customer ID", "StockCode"]]
)

test_pairs = pd.MultiIndex.from_frame(
    test[["Customer ID", "StockCode"]]
)

assert not test_pairs.isin(train_pairs).any()

assert test["Customer ID"].isin(
    train["Customer ID"]
).all()

print("\nAll DE3 validation checks passed.")

# Check cold-start products
train_products = set(train["StockCode"])

cold_start_products = ~test["StockCode"].isin(train_products)

print("\nCold-start product rows:", cold_start_products.sum())
print("Evaluable test rows:", (~cold_start_products).sum())

# 10. Save outputs
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train.to_pickle(output_dir / "train.pkl")
test.to_pickle(output_dir / "test.pkl")

# 11. Create split summary
# 11. Create split summary
split_summary = pd.DataFrame({
    "Metric": [
        "Original cleaned rows",
        "Training rows",
        "Test-period rows",
        "Held-out new-product rows",
        "Cold-start product rows",
        "Evaluable test rows",
        "Training customers",
        "Eligible test customers"
    ],
    "Value": [
        len(df),
        len(train),
        len(test_period),
        len(test),
        cold_start_products.sum(),
        (~cold_start_products).sum(),
        train["Customer ID"].nunique(),
        test["Customer ID"].nunique()
    ]
})

split_summary.to_csv(
    output_dir / "split_summary.csv",
    index=False
)

print("\nSplit Summary:")
print(split_summary.to_string(index=False))