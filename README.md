# MiniProject_LinearAlgebra

## Project Overview

This project uses the UCI Online Retail II dataset to build a customer-product recommendation system using dimensionality reduction techniques such as PCA and Truncated SVD.

The goal is to reduce the size of the customer-product matrix while maintaining useful information for product recommendations and customer segmentation.

### Project Objectives

- Build a clean and reliable customer-product dataset.
- Reduce data size using dimensionality reduction.
- Generate product recommendations based on customer purchasing patterns.
- Compare recommendation performance against a best-seller baseline.
- Visualize customer groups for marketing analysis.
- Evaluate model performance, storage requirements, and computational efficiency.

## Dataset

- **Source:** UCI Online Retail II
- **Dataset URL:** https://archive.ics.uci.edu/dataset/502/online+retail+ii
- **Period:** December 2009 – December 2011
- **Total Rows:** 1,067,371
- **Columns:** 8
- **Unique Customers:** 5,942
- **Unique Product Codes:** 5,305
- **Missing Customer IDs:** 243,007
- **Missing Descriptions:** 4,382

The dataset contains two Excel sheets:

- Year 2009-2010: 525,461 rows
- Year 2010-2011: 541,910 rows

Raw data is stored locally in `data/raw/` and excluded from Git through `.gitignore`.

## DE1: Data Loading and Profiling

Both Excel sheets were loaded using pandas and analyzed to identify data types, column names, missing values, and unique values.

### Data Profile

| Column | Data Type | Missing Values | Unique Values |
|---|---|---:|---:|
| Invoice | object | 0 | 53,628 |
| StockCode | object | 0 | 5,305 |
| Description | object | 4,382 | 5,698 |
| Quantity | int64 | 0 | 1,057 |
| InvoiceDate | datetime64[us] | 0 | 47,635 |
| Price | float64 | 0 | 2,807 |
| Customer ID | float64 | 243,007 | 5,942 |
| Country | str | 0 | 43 |

### Implementation

- **Script:** `data_profile.py`
- Loaded both Excel sheets.
- Verified row counts and column names.
- Examined data types and missing values.
- Generated a data profile table.

## DE2: Data Cleaning

Data cleaning was performed to remove invalid transactions and prepare the dataset for recommendation modeling.

### Cleaning Results

| Cleaning Rule | Rows Removed | Remaining Rows |
|---|---:|---:|
| Original Data | — | 1,067,371 |
| Cancelled Invoices | 19,494 | 1,047,877 |
| Non-positive Quantity | 3,457 | 1,044,420 |
| Non-positive Price | 2,750 | 1,041,670 |
| Non-product Stock Codes | 2,860 | 1,038,810 |
| Missing Customer ID | 235,890 | 802,920 |
| **Total Removed** | **264,451** | **802,920** |

### Cleaning Decisions

1. **Cancelled Invoices:** Removed transactions with invoice numbers starting with `C`, as they represent cancelled orders.
2. **Non-positive Quantity:** Removed rows where quantity was zero or negative.
3. **Non-positive Price:** Removed rows where price was zero or negative.
4. **Non-product Stock Codes:** Removed selected shipping, fee, discount, adjustment, and test codes that do not represent recommendable products.
5. **Missing Customer ID:** Removed transactions without customer identifiers because customer IDs are required to construct the customer-product matrix.

### Validation

The cleaned dataset was checked for:

- Cancelled invoices
- Non-positive quantities
- Non-positive prices
- Selected non-product stock codes
- Missing Customer IDs
- Missing Descriptions

All validation checks returned zero remaining invalid records under the implemented rules.

### Implementation

- **Script:** `data_cleaning.py`
- **Decision Log:** `Cleaning_decisions.md`
- **Cleaned Dataset:** `data/processed/cleaned_retail.pkl`
- **Cleaning Log:** `data/processed/cleaning_log.csv`

## DE3: Time-based Train/Test Split

The cleaned dataset was divided chronologically into training and test periods to prevent data leakage during model training.

### Split Strategy

- **Earliest Transaction:** December 1, 2009, 07:45
- **Latest Transaction:** December 9, 2011, 12:50
- **Test Period Start:** October 9, 2011, 12:50
- **Training Period:** Transactions before the test start
- **Test Period:** Transactions from the test start onward

The final two months of transaction history were reserved for testing.

For each customer, purchases of products not previously purchased during training were identified as held-out test purchases.

Customers without training history were excluded from the held-out recommendation evaluation.

### Split Results

| Metric | Value |
|---|---:|
| Cleaned Rows | 802,920 |
| Training Rows | 684,312 |
| Test-period Rows | 118,608 |
| Held-out New-product Rows | 56,919 |
| Cold-start Product Rows | 4,334 |
| Evaluable Test Rows | 52,585 |
| Training Customers | 5,475 |
| Eligible Test Customers | 1,930 |

### Validation

- Confirmed chronological separation between training and test periods.
- Verified that held-out customer-product pairs do not appear in training data.
- Verified that eligible test customers have training history.
- Identified cold-start products not present in training data.

All implemented DE3 validation checks passed.

The final evaluation set will be restricted to recommendable products after applying the training-based product filter.

### Implementation

- **Script:** `data_split.py`
- **Training Data:** `data/processed/train.pkl`
- **Held-out Test Data:** `data/processed/test.pkl`
- **Split Summary:** `data/processed/split_summary.csv`

## DE4: Sparse Customer-Product Matrix

A sparse customer-product matrix was constructed using training data only.

Each row represents a customer, each column represents a product, and each matrix value represents the customer's purchasing activity for that product.

### Matrix Construction

- **Purchase Representation:** `log(1 + total quantity purchased per customer-product pair)`
- **Product Filtering:** Products purchased by at least 10 unique customers
- **Matrix Format:** Compressed Sparse Row (CSR)
- **Training Rows:** 684,312
- **Filtered Training Rows:** 679,944

### Results

| Metric | Value |
|---|---:|
| Customers | 5,475 |
| Products | 3,689 |
| Matrix Shape | 5,475 × 3,689 |
| Non-zero Entries | 413,983 |
| Missing Customer Indices | 0 |
| Missing Product Indices | 0 |
| Duplicate Customer-Product Pairs | 0 |

### Design Decisions

**1. Log-transformed purchase quantities**

Log-transformed purchase quantities were selected to reduce the influence of unusually large orders while preserving differences in purchasing volume.

**2. Product Filtering**

Products purchased by fewer than 10 unique customers were excluded to reduce noise from infrequently purchased products.

**3. Sparse Matrix Representation**

The CSR format was selected because most customer-product combinations contain zero purchases. Sparse storage avoids storing these zeros explicitly.

**4. Data Leakage Prevention**

Only training data was used to construct the matrix. Held-out purchases were excluded from the matrix construction process.

### Validation

- Confirmed matrix dimensions.
- Verified customer and product index mappings.
- Verified that no matrix indices were missing.
- Verified that customer-product combinations were unique.
- Confirmed that the number of non-zero matrix entries matched the number of aggregated customer-product pairs.

All DE4 matrix validation checks passed.

### Implementation

- **Script:** `build_matrix.py`
- **Sparse Matrix:** `data/processed/customer_product_matrix.npz`
- **Customer Index Map:** `data/processed/customer_map.pkl`
- **Product Index Map:** `data/processed/product_map.pkl`

## Project Structure

```text
MiniProject_LinearAlgebra/
├── data/
│   ├── raw/
│   │   └── online_retail_II.xlsx
│   └── processed/
│       ├── cleaned_retail.pkl
│       ├── cleaning_log.csv
│       ├── train.pkl
│       ├── test.pkl
│       ├── split_summary.csv
│       ├── customer_product_matrix.npz
│       ├── customer_map.pkl
│       └── product_map.pkl
├── miniproject/
├── .gitignore
├── README.md
├── requirements.txt
├── data_profile.py
├── data_cleaning.py
├── data_split.py
├── build_matrix.py
└── Cleaning_decisions.md
```

The `data/` directory and `miniproject/` virtual environment are excluded from Git.

## Installation and Reproduction

### 1. Clone the Repository

```bash
git clone https://github.com/chrismskim/MiniProject_LinearAlgebra.git
cd MiniProject_LinearAlgebra
```

### 2. Create a Python Virtual Environment

```bash
python -m venv miniproject
source miniproject/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Download the Dataset

Download Online Retail II from:

https://archive.ics.uci.edu/dataset/502/online+retail+ii

Extract `online_retail_II.xlsx` and place it in:

`data/raw/online_retail_II.xlsx`

### 5. Run the Data Pipeline

Execute the scripts in the following order:

```bash
python data_profile.py
python data_cleaning.py
python data_split.py
python build_matrix.py
```

The scripts generate the cleaned dataset, training and test datasets, sparse customer-product matrix, and associated index mappings.

## Reproducibility

- Dependencies are recorded in `requirements.txt`.
- Training and test data are separated chronologically before modeling.
- Matrix construction uses training data only.
- Data cleaning and processing steps are documented.
- Generated data files are excluded from Git and can be recreated using the pipeline.

Random seeds will be fixed when stochastic modeling methods are introduced.

## Next Steps

- **DE5:** Analyze matrix shape, sparsity, and memory usage.
- **DE6:** Package data processing into a reproducible pipeline.
- **ML1–ML7:** Perform dimensionality reduction, recommendation modeling, evaluation, and customer segmentation.
- **J2:** Prepare the stakeholder memo.
- **J3:** Prepare the final presentation.
## DE5: Matrix Statistics and Memory Report

The customer-product matrix was analyzed to measure sparsity and compare dense versus sparse storage requirements.

### Results

| Metric | Value |
|---|---:|
| Matrix Shape | 5,475 × 3,689 |
| Total Entries | 20,197,275 |
| Non-zero Entries | 413,983 |
| Sparsity | 97.95% |
| Density | 2.05% |
| Dense Memory | 154.09 MiB |
| Sparse Memory (CSR) | 4.76 MiB |
| Memory Reduction | 96.91% |
| Compression Ratio | 32.38× |

### Findings

- Approximately 97.95% of matrix entries are zero.
- CSR storage reduces memory usage from 154.09 MiB to 4.76 MiB.
- Sparse storage requires approximately 32.38 times less memory than dense storage.
- These measurements compare storage formats before PCA or SVD dimensionality reduction.

### Implementation

- **Script:** `matrix_report.py`
- **Output:** `reports/matrix_report.csv`

All implemented matrix validation checks passed.


## DE6: Reproducible Data Pipeline

The complete data engineering pipeline was automated to rebuild the customer-product matrix directly from the raw dataset using a single command.

### Execution

Run the following command from the project root directory:

```bash
python build_matrix.py
```

### Pipeline Steps

1. **Data Cleaning (DE2):** Loads the raw Excel dataset, removes invalid transactions, validates the results, and saves the cleaned data.
2. **Train/Test Split (DE3):** Splits the cleaned dataset chronologically, identifies held-out purchases, and saves the training and test datasets.
3. **Sparse Matrix Construction (DE4):** Builds the sparse customer-product matrix using training data only and saves the index mappings.

### Execution Results

| Metric | Value |
|