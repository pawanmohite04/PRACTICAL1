import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("house_data (2).csv")

print("First 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# 2. Select features (X) and target (y)
features = ['sqft_living', 'grade', 'bathrooms',
            'sqft_above', 'sqft_living15', 'view', 'bedrooms']

X = df[features]
y = df['price']

# 3. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Coefficients & intercept
print("\nIntercept (b0):", model.intercept_)
print("Coefficients:")
for name, coef in zip(features, model.coef_):
    print(f"  {name}: {coef}")

print("\nModel Equation:")
print(f"price = {model.intercept_:.2f}"
      f" + {model.coef_[0]:.2f} * sqft_living"
      f" + {model.coef_[1]:.2f} * grade"
      f" + {model.coef_[2]:.2f} * bathrooms"
      f" + {model.coef_[3]:.2f} * sqft_above"
      f" + {model.coef_[4]:.2f} * sqft_living15"
      f" + {model.coef_[5]:.2f} * view"
      f" + {model.coef_[6]:.2f} * bedrooms"
)

# 6. Predictions & evaluation
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("\nTrain MSE:", mse_train)
print("Test MSE:", mse_test)
print("Train RMSE:", np.sqrt(mse_train))
print("Test RMSE:", np.sqrt(mse_test))
print("Train R²:", r2_train)
print("Test R²:", r2_test)

# 7. Plot: Predicted vs Actual (Test set)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5)
max_val = max(y_test.max(), y_test_pred.max())
min_val = min(y_test.min(), y_test_pred.min())
plt.plot([min_val, max_val], [min_val, max_val], linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual House Prices (Test Set)")
plt.grid(True)
plt.show()

# 8. Optional: Residual plot
residuals = y_test - y_test_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_test_pred, residuals, alpha=0.5)
plt.axhline(0, linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()

