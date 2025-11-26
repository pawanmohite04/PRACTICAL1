# ---------------------------------------------------------
# Assignment 8 – Unsupervised Model Choice: Credit Card Fraud Detection
# Algorithm: Isolation Forest (Unsupervised Anomaly Detection)
# ---------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

# ---------------------------------------------------------
# 1. Load Dataset
# ---------------------------------------------------------
df = pd.read_csv("creditcard.csv")
print("Dataset Shape:", df.shape)
print(df.head())

# ---------------------------------------------------------
# 2. Dataset Description (Counts)
# ---------------------------------------------------------
print("\nClass Distribution:")
print(df["Class"].value_counts())

fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0]

print("\nFraud Cases:", len(fraud))
print("Normal Cases:", len(normal))
print("Fraud Percentage: {:.4f}%".format((len(fraud) / len(df)) * 100))

# ---------------------------------------------------------
# 3. Preprocessing – Standard Scaling
# ---------------------------------------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------
# 4. Isolation Forest (Unsupervised)
# ---------------------------------------------------------
iso = IsolationForest(
    n_estimators=200,
    contamination=0.0017,   # approx fraud rate (0.17%)
    max_samples='auto',
    random_state=42
)

iso.fit(X_scaled)

# Predictions: -1 = anomaly, 1 = normal
df["Pred"] = iso.predict(X_scaled)
df["Pred"] = df["Pred"].apply(lambda x: 1 if x == -1 else 0)

# ---------------------------------------------------------
# 5. Evaluation
# ---------------------------------------------------------
y_true = y
y_pred = df["Pred"]

print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))

# Heatmap of confusion matrix
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Isolation Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------------------------------------
# 6. Visualization of Anomalies (Fraud vs Normal)
# ---------------------------------------------------------
plt.figure(figsize=(8,6))
plt.scatter(df["Time"][:3000], df["Amount"][:3000],
            c=df["Pred"][:3000], cmap="coolwarm", s=12)
plt.title("Fraud Detection Visualization (Isolation Forest)")
plt.xlabel("Time")
plt.ylabel("Transaction Amount")
plt.show()

# ---------------------------------------------------------
# END OF CODE
# ---------------------------------------------------------
