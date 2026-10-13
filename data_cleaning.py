
import pandas as pd
from pathlib import Path

# DE2: Data Cleaning

# 1. Load both Excel sheets
sheets = pd.read_excel(
    "data/raw/online_retail_II.xlsx",
    sheet_name=None
)

df = pd.concat(sheets.values(), ignore_index=True)

original_rows = len(df)  #Use for removed rows later


# 2. Define non-product codes
non_product_codes = [
    "POST", "M", "D", "BANK CHARGES",    #Delete the non sale code
    "AMAZONFEE", "ADJUST", "ADJUST2",
    "TEST001", "TEST002"
]

# 3. Clean transactions and record removed rows
cleaning_results = []

def remove_rows(rule_name, invalid_mask):    #Delete which one, condition for deleting 
    global df #Allow reassignment of the global df

    removed = int(invalid_mask.sum())   #count the number of deleting row
    df = df.loc[~invalid_mask].copy()   # Keep only valid rows

    #Save Cleaning Log
    cleaning_results.append({
        "Cleaning Rule": rule_name,
        "Rows Removed": removed
    })

    print(f"{rule_name}: removed {removed} rows")


#Data cleaning
remove_rows(
    "Cancelled Invoices",
    df["Invoice"].astype(str).str.startswith("C")
)

remove_rows(
    "Non-positive Quantity",
    df["Quantity"] <= 0
)

remove_rows(
    "Non-positive Price",
    df["Price"] <= 0
)

remove_rows(
    "Non-product Stock Codes",
    df["StockCode"].astype(str).str.strip().str.upper()
    .isin(non_product_codes)
)

remove_rows(
    "Missing Customer ID",
    df["Customer ID"].isna()
)

# 4. Create cleaning log
cleaning_log = pd.DataFrame(cleaning_results)

# 5. Save cleaned data
output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

df.to_pickle(output_dir / "cleaned_retail.pkl")   #Pickle is usefull to get datas from Dataframe

cleaning_log.to_csv(
    output_dir / "cleaning_log.csv",
    index=False
)

