# ----------------------------------------
# PCA on House Pricing Dataset
# ----------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("house_data (2).csv")

# Select numeric columns
num_df = df.select_dtypes(include=['int64','float64'])

# Standardize
scaler = StandardScaler()
scaled = scaler.fit_transform(num_df)

# PCA full decomposition
pca = PCA()
pca_data = pca.fit_transform(scaled)

# Plot explained variance
plt.figure(figsize=(7,4))
plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title("Cumulative Explained Variance – House Prices")
plt.xlabel("Number of Components")
plt.ylabel("Variance Explained")
plt.grid()
plt.show()

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Keep components that explain 95% variance
pca_95 = PCA(n_components=0.95)
hp_pca = pca_95.fit_transform(scaled)

print("\nShape after PCA (95% variance):", hp_pca.shape)
