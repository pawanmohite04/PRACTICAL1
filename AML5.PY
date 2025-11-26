# ---------------------------------------------
# Assignment 5 – Multi-Class Classification
# Wine Dataset (Your file: wine-class (1).csv)
# ---------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------
# 1. Load Dataset
# ---------------------------------------------
df = pd.read_csv("wine-class (1).csv")

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nCLASS COUNTS:")
print(df['class'].value_counts())

# ---------------------------------------------
# 2. Train-Test Split (80/20, stratified)
# ---------------------------------------------
X = df.drop("class", axis=1)
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------
# 3. Feature Scaling
# ---------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------
# 4. Handle Imbalance Using SMOTE
# ---------------------------------------------
print("\nBEFORE SMOTE:", y_train.value_counts())

sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train)

print("\nAFTER SMOTE:", y_train_smote.value_counts())

# ---------------------------------------------
# 5. Model 1 – Logistic Regression
# ---------------------------------------------
log_reg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=2000,
    class_weight="balanced"
)
log_reg.fit(X_train_smote, y_train_smote)

y_pred_log = log_reg.predict(X_test_scaled)

print("\n================ Logistic Regression ================")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_log))

print("Accuracy:", accuracy_score(y_test, y_pred_log))

# ---------------------------------------------
# 6. Model 2 – Random Forest Classifier
# ---------------------------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

print("\n================ Random Forest =================")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

print("Accuracy:", accuracy_score(y_test, y_pred_rf))

# ---------------------------------------------
# 7. Confusion Matrix Heatmap (Random Forest)
# ---------------------------------------------
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
