
import pandas as pd
from pathlib import Path

# DE3: Time-based Train/Test Split

# 1. Load cleaned dataset
df = pd.read_pickle("data/processed/cleaned_retail.pkl")

# Normalize product codes
df["StockCode"] = df["StockCode"].astype(str).str.strip()

# 2. Define the last two months as the test period
latest_date = df["InvoiceDate"].max()
test_start = latest_date - pd.DateOffset(months=2)

# 3. Split data by time
train = df[df["InvoiceDate"] < test_start].copy()
test_period = df[df["InvoiceDate"] >= test_start].copy()

# 4. Identify products already purchased by each customer
previous_purchases = (
    train[["Customer ID", "StockCode"]]
    .drop_duplicates()  # Remove duplicate customer-product pairs
    .assign(purchased_before=True)
)

# 5. Match test purchases against training history
test_candidates = test_period.merge(
    previous_purchases,
    on=["Customer ID", "StockCode"],
    how="left",
    validate="many_to_one"
)

# 6. Keep only previously unpurchased products
test = test_candidates[
    test_candidates["purchased_before"].isna()
].drop(columns=["purchased_before"]).copy()

# 7. Keep only customers from training
train_customers = set(train["Customer ID"])

test = test[
    test["Customer ID"].isin(train_customers)
].copy()

# 8. Check cold-start products
train_products = set(train["StockCode"])

cold_start_products = ~test["StockCode"].isin(train_products)

# 9. Basic validation
assert len(train) + len(test_period) == len(df)
assert test["Customer ID"].isin(train_customers).all()

# Make sure test purchases were not in training
assert not test.merge(
    previous_purchases,
    on=["Customer ID", "StockCode"],
    how="inner"
).shape[0]

# 10. Save outputs
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train.to_pickle(output_dir / "train.pkl")
test.to_pickle(output_dir / "test.pkl")

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
        int(cold_start_products.sum()),
        int((~cold_start_products).sum()),
        train["Customer ID"].nunique(),
        test["Customer ID"].nunique()
    ]
})

split_summary.to_csv(
    output_dir / "split_summary.csv",
    index=False
)

print(split_summary.to_string(index=False))

