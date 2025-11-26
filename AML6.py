# ----------------------------------------------------
# Assignment 6 – K-Means Clustering (Customer Segmentation)
# Dataset: Mall_Customers.csv
# ----------------------------------------------------

# FIX WARNING (Add this at the top)
import os
os.environ["OMP_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

# ----------------------------------------------------
# 1. Load Dataset
# ----------------------------------------------------
df = pd.read_csv("Mall_Customers.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(df.columns)

# ----------------------------------------------------
# 2. Select Features for Clustering
# ----------------------------------------------------
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

print("\nSelected Features:")
print(X.head())

# ----------------------------------------------------
# 3. Scaling the Features
# ----------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ----------------------------------------------------
# 4. Elbow Method to find Optimal k
# ----------------------------------------------------
wcss = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)   # inertia = WCSS

plt.figure(figsize=(7, 4))
plt.plot(K, wcss, marker='o')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()

# ----------------------------------------------------
# 5. Silhouette Score Method
# ----------------------------------------------------
silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.figure(figsize=(7, 4))
plt.plot(range(2, 11), silhouette_scores, marker='o', color='orange')
plt.title("Silhouette Scores for Different k")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.show()

print("\nSilhouette Scores (k=2..10):")
for k, s in zip(range(2, 11), silhouette_scores):
    print(f"k={k}: {s:.4f}")

# ----------------------------------------------------
# 6. Apply K-Means Clustering using Optimal k
# ----------------------------------------------------
optimal_k = np.argmax(silhouette_scores) + 2   # because silhouette starts from k=2
print("\nOptimal clusters chosen =", optimal_k)

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
cluster_labels = kmeans_final.fit_predict(X_scaled)

df['Cluster'] = cluster_labels

print("\nCluster counts:")
print(df['Cluster'].value_counts())

# ----------------------------------------------------
# 7. Visualize Clusters (2D pairplot)
# ----------------------------------------------------
sns.pairplot(df, vars=['Age','Annual Income (k$)','Spending Score (1-100)'],
             hue='Cluster', palette='tab10')
plt.suptitle("Customer Segmentation Clusters", y=1.02)
plt.show()

# ----------------------------------------------------
# 8. Final Cluster Centers (scaled → original)
# ----------------------------------------------------
centers_scaled = kmeans_final.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)

print("\nCluster Centers (Original Scale):")
center_df = pd.DataFrame(centers_original,
                         columns=['Age','Annual Income (k$)','Spending Score (1-100)'])
print(center_df)
