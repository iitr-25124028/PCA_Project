import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Load the datasets
# Pandas can read .gz files directly
expr = pd.read_csv("filtered.tsv.gz", sep="\t")
expr.columns = expr.columns.str.strip()

# Labels for the 105 patients
classes = pd.read_csv("class.tsv", sep="\t", header=None)

# Gene mapping file
cols = pd.read_csv(
    "columns.tsv.gz", 
    sep="\t", 
    comment="#", 
    header=None, 
    on_bad_lines='skip'
)

# 2. Identify the Gene IDs for XBP1 and GATA3
# We look for the gene symbol in column index 4 and get the ID from column index 0
xbp1_id = str(int(cols[cols[4].astype(str).str.upper() == "XBP1"].iloc[0,0]))
gata3_id = str(int(cols[cols[4].astype(str).str.upper() == "GATA3"].iloc[0,0]))

# 3. Figure 1a: Gene vs Gene Scatter Plot (105 points)
plt.figure(figsize=(8,6))
plt.scatter(expr[xbp1_id], expr[gata3_id], c=classes[0], cmap='viridis', alpha=0.8)
plt.xlabel(f"XBP1 Expression (ID: {xbp1_id})")
plt.ylabel(f"GATA3 Expression (ID: {gata3_id})")
plt.title("Figure 1a: XBP1 vs GATA3 Scatter Plot")
plt.colorbar(label='Class (0:ER-, 1:ER+)')
plt.savefig("scatter_plot.png")
plt.show()

# 4. PCA for Patient Clustering
# PCA expects (n_samples, n_features). 
# Since expr is (105 patients, 16174 genes), it is already in the correct format.
# We remove the .T (transpose) that caused the 16174 vs 105 error.
X = expr.select_dtypes(include=['number'])

# Step 4.1: Scaling (Crucial for biological data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4.2: Run PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Figure 1c: PCA Plot (105 points)
plt.figure(figsize=(8,6))
# This now plots 105 dots, matching your 105 class labels
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=classes[0], cmap='viridis', alpha=0.8)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
plt.title("Figure 1c: PCA Clustering of Patients")
plt.colorbar(label='Class (0:ER-, 1:ER+)')
plt.savefig("pca_plot_final.png")
plt.show()

print(f"PCA complete. PC1 explains {pca.explained_variance_ratio_[0]:.2%} of the variance.")
