import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# -------------------------
# Load data
# -------------------------

expr = pd.read_csv("filtered.tsv.gz", sep="\t")

# remove spaces from column names
expr.columns = expr.columns.str.strip()

classes = pd.read_csv(
    "class.tsv",
    sep="\t",
    header=None
)

cols = pd.read_csv(
    "columns.tsv.gz",
    sep="\t",
    comment="#",
    header=None,
    on_bad_lines='skip'
)

# -------------------------
# Debug expression matrix
# -------------------------

print("\nExpression Data:")
print(expr.head())

print("\nExpression Columns:")
print(expr.columns[:20])

# -------------------------
# Find XBP1
# -------------------------

xbp1_row = cols[
    cols[4].astype(str).str.upper() == "XBP1"
]

print("\nXBP1 ROW:")
print(xbp1_row)

xbp1_id = int(xbp1_row.iloc[0,0])

# -------------------------
# Find GATA3
# -------------------------

gata3_row = cols[
    cols[4].astype(str).str.upper() == "GATA3"
]

print("\nGATA3 ROW:")
print(gata3_row)

gata3_id = int(gata3_row.iloc[0,0])

print("\nXBP1 ID:", xbp1_id)
print("GATA3 ID:", gata3_id)

# -------------------------
# Extract expression
# -------------------------

xbp1 = expr[str(xbp1_id)]
gata3 = expr[str(gata3_id)]

# -------------------------
# Scatter Plot
# -------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    xbp1,
    gata3,
    c=classes[0]
)

plt.xlabel("XBP1")
plt.ylabel("GATA3")
plt.title("XBP1 vs GATA3")

plt.savefig("scatter_plot.png")

plt.show()

# -------------------------
# PCA
# -------------------------

# Keep only numeric columns
expr_numeric = expr.select_dtypes(include=['number'])

# transpose matrix
X = expr_numeric.T

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X)

# -------------------------
# PCA Plot
# -------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=classes[0]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA on Gene Expression Data")

plt.savefig("pca_plot.png")

plt.show()

print("\nDONE!")