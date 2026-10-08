# DE2 — Data Cleaning Decision Log

## Dataset Summary

- Original rows: 1,067,371
- Removed rows: 264,451
- Remaining rows: 802,920

## Cleaning Decisions

| Rule | Rows Removed | Reason |
|---|---:|---|
| Cancelled Invoices | 19,494 | Cancelled transactions do not represent completed purchases. |
| Non-positive Quantity | 3,457 | Zero or negative quantities do not represent valid positive purchases. |
| Non-positive Price | 2,750 | Non-positive prices are excluded from purchase records. |
| Non-product Stock Codes | 2,860 | Exclude selected shipping, fee, adjustment, discount, and test codes that do not represent recommendable products. |
| Missing Customer ID | 235,890 | Customer identifiers are required to construct the customer-product matrix. |

## Validation

All five cleaning rules were checked after processing. No remaining rows violated these rules.

The cleaned dataset also contains no missing product descriptions.

## Output

- `data/processed/cleaned_retail.pkl`
- `data/processed/cleaning_log.csv`

Raw and processed datasets are stored locally and excluded from Git.