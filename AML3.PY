import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("house_data (2).csv")

# Select features
features = ['sqft_living', 'grade', 'bathrooms',
            'sqft_above', 'sqft_living15', 'view', 'bedrooms']

X = df[features]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 1. Linear Regression
# ------------------------------
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

print("\n--- Linear Regression ---")
print("Test MSE:", mean_squared_error(y_test, y_pred_lin))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_lin)))
print("Test R²:", r2_score(y_test, y_pred_lin))

# ------------------------------
# 2. Random Forest Regression
# ------------------------------
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Regression ---")
print("Test MSE:", mean_squared_error(y_test, y_pred_rf))
print("Test RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_rf)))
print("Test R²:", r2_score(y_test, y_pred_rf))

# ------------------------------
# Plot (Random Forest)
# ------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_rf, alpha=0.4)
plt.plot([0, y_test.max()], [0, y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (Random Forest)")
plt.grid(True)
plt.show()
