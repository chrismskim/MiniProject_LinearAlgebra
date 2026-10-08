
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import pickle

# DE4: Build Sparse Customer-Product Matrix

# 1. Load training data only
train = pd.read_pickle("data/processed/train.pkl")

print("Training rows:", len(train))

# 2. Filter products purchased by at least 10 customers
product_customers = train.groupby("StockCode")["Customer ID"].nunique()

valid_products = product_customers[
    product_customers >= 10
].index

train_filtered = train[
    train["StockCode"].isin(valid_products)
].copy()

print("Filtered training rows:", len(train_filtered))

# 3. Create customer and product index mappings
# Normalize product codes before mapping
train_filtered["StockCode"] = (
    train_filtered["StockCode"]
    .astype(str)
    .str.strip()
)
customers = sorted(train["Customer ID"].unique())
products = sorted(train_filtered["StockCode"].astype(str).unique())

customer_map = {customer: i for i, customer in enumerate(customers)}
product_map = {product: j for j, product in enumerate(products)}

# 4. Aggregate quantities per customer-product pair
purchases = (
    train_filtered
    .groupby(["Customer ID", "StockCode"], as_index=False)["Quantity"]
    .sum()
)
purchases["StockCode"] = purchases["StockCode"].astype(str).str.strip()

# 5. Apply log(1 + quantity)
purchases["Value"] = np.log1p(purchases["Quantity"])

# 6. Convert IDs to matrix indices
rows = purchases["Customer ID"].map(customer_map).to_numpy()
cols = purchases["StockCode"].map(product_map).to_numpy()
values = purchases["Value"].to_numpy()

assert pd.notna(rows).all(), "Missing customer indices!"
assert pd.notna(cols).all(), "Missing product indices!"

# Check matrix indices
print("\nIndex Validation:")
print("Purchase pairs:", len(purchases))
print("Missing customer indices:", pd.isna(rows).sum())
print("Missing product indices:", pd.isna(cols).sum())

print("Duplicate customer-product pairs:",
      purchases.duplicated(
          subset=["Customer ID", "StockCode"]
      ).sum())

# 7. Build sparse matrix
A = sp.csr_matrix(
    (values, (rows, cols)),
    shape=(len(customers), len(products))
)

print("\nMatrix Shape:", A.shape)
print("Non-zero Values:", A.nnz)

# 8. Validate matrix
assert A.shape[0] == len(customers)
assert A.shape[1] == len(products)
assert A.nnz == len(purchases)
assert A.data.min() > 0

print("Matrix validation passed.")

# 9. Save outputs
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

sp.save_npz(output_dir / "customer_product_matrix.npz", A)

with open(output_dir / "customer_map.pkl", "wb") as f:
    pickle.dump(customer_map, f)

with open(output_dir / "product_map.pkl", "wb") as f:
    pickle.dump(product_map, f)

print("\nSparse matrix and index mappings saved.")
