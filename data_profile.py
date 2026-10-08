import pandas as pd

sheets = pd.read_excel(
    "data/raw/online_retail_II.xlsx",
    sheet_name=None
)

# Check each sheet
for name, sheet in sheets.items():
    print(f"Sheet: {name}")
    print(f"Rows: {sheet.shape[0]}")
    print(f"Columns: {sheet.shape[1]}")

    print("\nColumn Names:")
    print(sheet.columns.tolist())

    print("\nData Types:")
    print(sheet.dtypes)

    print("\nMissing Values:")
    print(sheet.isnull().sum())
    print()

# Outside the for loop
df = pd.concat(sheets.values(), ignore_index=True)

profile = pd.DataFrame({
    "Data Type": df.dtypes.astype(str),
    "Missing Values": df.isnull().sum(),
    "Unique Values": df.nunique(dropna=True)
})

print("\nData Profile Table:")
print(profile)