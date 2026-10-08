import pandas as pd
from pathlib import Path

# Load both sheets
sheets = pd.read_excel(
    "data/raw/online_retail_II.xlsx",
    sheet_name=None
)

df = pd.concat(sheets.values(), ignore_index=True)

# Original row count
print("Original rows:", len(df))

# Remove cancelled invoices
cancelled = df["Invoice"].astype(str).str.startswith("C")

removed = cancelled.sum()

df = df[~cancelled].copy()

print("Cancelled invoice rows removed:", removed)
print("Remaining rows:", len(df))


# Remove non-positive quantities
invalid_quantity = df["Quantity"] <= 0

removed_quantity = invalid_quantity.sum()

df = df[~invalid_quantity].copy()

print("Non-positive quantity rows removed:", removed_quantity)
print("Remaining rows:", len(df))


# Remove non-positive prices
invalid_price = df["Price"] <= 0

removed_price = invalid_price.sum()

df = df[~invalid_price].copy()

print("Non-positive price rows removed:", removed_price)
print("Remaining rows:", len(df))

# Remove non-product stock codes
non_product_codes = [
    "POST",
    "M",
    "D",
    "BANK CHARGES",
    "AMAZONFEE",
    "ADJUST",
    "ADJUST2",
    "TEST001",
    "TEST002"
]

stock_codes = df["StockCode"].astype(str).str.strip().str.upper()

invalid_codes = stock_codes.isin(non_product_codes)

removed_codes = invalid_codes.sum()

df = df[~invalid_codes].copy()

print("Non-product rows removed:", removed_codes)
print("Remaining rows:", len(df))

# Remove rows with missing Customer IDs
missing_customer = df["Customer ID"].isna()

removed_customer = missing_customer.sum()

df = df[~missing_customer].copy()

print("Missing Customer ID rows removed:", removed_customer)
print("Remaining rows:", len(df))


# Create cleaning log
cleaning_log = pd.DataFrame({
    "Cleaning Rule": [
        "Cancelled Invoices",
        "Non-positive Quantity",
        "Non-positive Price",
        "Non-product Stock Codes",
        "Missing Customer ID"
    ],
    "Rows Removed": [
        removed,
        removed_quantity,
        removed_price,
        removed_codes,
        removed_customer
    ]
})

print("\nCleaning Log:")
print(cleaning_log)

print("\nTotal rows removed:", cleaning_log["Rows Removed"].sum())
print("Final rows:", len(df))


# Validate cleaned data
print("\nData Validation:")

print("Cancelled invoices:",
      df["Invoice"].astype(str).str.startswith("C").sum())

print("Non-positive quantities:",
      (df["Quantity"] <= 0).sum())

print("Non-positive prices:",
      (df["Price"] <= 0).sum())

print("Non-product codes:",
      df["StockCode"].astype(str).str.strip().str.upper()
      .isin(non_product_codes).sum())

print("Missing Customer IDs:",
      df["Customer ID"].isna().sum())

# Check remaining missing descriptions
print("\nRemaining missing descriptions:")
print(df["Description"].isna().sum())


# Create output directory
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Save cleaned dataset
df.to_pickle(output_dir / "cleaned_retail.pkl")

# Save cleaning log
cleaning_log.to_csv(
    output_dir / "cleaning_log.csv",
    index=False
)

print("\nCleaned dataset and cleaning log saved.")