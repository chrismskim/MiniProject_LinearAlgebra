
import numpy as np
import scipy.sparse as sp
import pandas as pd
from pathlib import Path

# DE5: Matrix Statistics and Memory Report

# 1. Load sparse matrix
A = sp.load_npz("data/processed/customer_product_matrix.npz")

# 2. Matrix dimensions
rows, cols = A.shape
total_entries = rows * cols
non_zero = A.nnz

# 3. Calculate sparsity
sparsity = (1 - non_zero / total_entries) * 100
density = (non_zero / total_entries) * 100

# 4. Calculate dense memory usage
dense_memory = total_entries * A.dtype.itemsize

# 5. Calculate sparse memory usage (CSR)
sparse_memory = (
    A.data.nbytes
    + A.indices.nbytes
    + A.indptr.nbytes
)

# 6. Convert bytes to MiB
dense_mb = dense_memory / (1024 ** 2)
sparse_mb = sparse_memory / (1024 ** 2)

# 7. Calculate memory reduction
reduction = (1 - sparse_memory / dense_memory) * 100
compression_ratio = dense_memory / sparse_memory

# 8. Display results
print("\nDE5: Matrix Statistics and Memory Report")
print("--------------------------------------")
print("Matrix Shape:", A.shape)
print("Total Entries:", total_entries)
print("Non-zero Entries:", non_zero)
print(f"Sparsity: {sparsity:.2f}%")
print(f"Density: {density:.2f}%")
print(f"Dense Memory: {dense_mb:.2f} MiB")
print(f"Sparse Memory: {sparse_mb:.2f} MiB")
print(f"Memory Reduction: {reduction:.2f}%")
print(f"Compression Ratio: {compression_ratio:.2f}x")

# 9. Validation
assert sp.isspmatrix_csr(A), "Matrix must be CSR"
assert A.nnz <= total_entries
assert 0 <= sparsity <= 100
assert sparse_memory < dense_memory

print("\nMatrix report validation passed.")

# 10. Save report
report = pd.DataFrame({
    "Metric": [
        "Customers",
        "Products",
        "Total Entries",
        "Non-zero Entries",
        "Sparsity (%)",
        "Density (%)",
        "Dense Memory (MiB)",
        "Sparse Memory (MiB)",
        "Memory Reduction (%)",
        "Compression Ratio"
    ],
    "Value": [
        rows,
        cols,
        total_entries,
        non_zero,
        round(sparsity, 2),
        round(density, 2),
        round(dense_mb, 2),
        round(sparse_mb, 2),
        round(reduction, 2),
        round(compression_ratio, 2)
    ]
})

output_dir = Path("reports")
output_dir.mkdir(parents=True, exist_ok=True)

report.to_csv(
    output_dir / "matrix_report.csv",
    index=False
)

print("\nReport saved to reports/matrix_report.csv")
