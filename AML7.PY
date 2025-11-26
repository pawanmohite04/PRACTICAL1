# ---------------------------------------------------------
# Assignment 7 – PCA + Clustering Pipeline (Movies Dataset)
# ---------------------------------------------------------

# ---------------------------------------------------------
# Assignment 7 – PCA + Clustering Pipeline (Movies Dataset)
# ---------------------------------------------------------

# MUST BE FIRST - Fix OpenMP warning before sklearn loads
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# Now import other libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("movies.csv")
print("First 5 rows:")
print(df.head())

# Select numeric columns only
num_df = df.select_dtypes(include=['int64', 'float64'])
print("\nNumeric columns used:")
print(num_df.columns)

# -----------------------------
# 2. Standardization
# -----------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(num_df)

# -----------------------------
# 3. Apply PCA (retain 95% variance)
# -----------------------------
pca = PCA(n_components=0.95)
pca_data = pca.fit_transform(scaled_data)

print("\nPCA shape:", pca_data.shape)
print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

# -----------------------------
# 4. Elbow Method after PCA
# -----------------------------
wcss = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pca_data)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(2, 11), wcss, marker='o')
plt.title("Elbow Method (After PCA)")
plt.xlabel("k")
plt.ylabel("WCSS")
plt.grid()
plt.show()

# -----------------------------
# 5. Silhouette Method
# -----------------------------
sil_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(pca_data)
    sil_scores.append(silhouette_score(pca_data, labels))

plt.figure(figsize=(6,4))
plt.plot(range(2, 11), sil_scores, marker='o', color="orange")
plt.title("Silhouette Scores (After PCA)")
plt.xlabel("k")
plt.ylabel("Score")
plt.grid()
plt.show()

best_k = np.argmax(sil_scores) + 2
print("\nBest k =", best_k)

# -----------------------------
# 6. Final K-Means with best k
# -----------------------------
final_kmeans = KMeans(n_clusters=best_k, random_state=42)
df["Cluster"] = final_kmeans.fit_predict(pca_data)

print("\nCluster counts:")
print(df["Cluster"].value_counts())

# -----------------------------
# 7. Visualize Clusters in PCA 2D
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(pca_data[:,0], pca_data[:,1], c=df["Cluster"], cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Movie Clusters After PCA")
plt.colorbar()
plt.show()
