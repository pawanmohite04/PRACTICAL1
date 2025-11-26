# --------------------------------------------
# STEP 1: Import Libraries
# --------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------
# STEP 2: Load Dataset
# --------------------------------------------
df = pd.read_csv("SOCR-HeightWeight.csv")

print("First 5 rows of dataset:")
print(df.head())


# --------------------------------------------
# STEP 3: Check Missing Values
# --------------------------------------------
print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------
# STEP 4: Select Feature & Target
# --------------------------------------------
X = df[['Height(Inches)']]
y = df['Weight(Pounds)']


# --------------------------------------------
# STEP 5: Train–Test Split
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------
# STEP 6: Train Linear Regression Model
# --------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\nIntercept (b0):", model.intercept_)
print("Coefficient (b1):", model.coef_[0])
print("\nModel Equation:")
print(f"Weight = {model.intercept_:.2f} + {model.coef_[0]:.2f} * Height")


# --------------------------------------------
# STEP 7: Predictions & Evaluation
# --------------------------------------------
y_train_pred = model.predict(X_train)   # using DataFrame → no warning
y_test_pred = model.predict(X_test)     # using DataFrame → no warning

print("\nTrain MSE:", mean_squared_error(y_train, y_train_pred))
print("Test MSE:", mean_squared_error(y_test, y_test_pred))
print("Train R²:", r2_score(y_train, y_train_pred))
print("Test R²:", r2_score(y_test, y_test_pred))


# --------------------------------------------
# STEP 8: Scatter Plot + Regression Line
# --------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.5, label="Actual Data")

# Regression line
line_x = np.linspace(df['Height(Inches)'].min(), df['Height(Inches)'].max(), 100)
line_y = model.predict(pd.DataFrame(line_x, columns=['Height(Inches)']))

plt.plot(line_x, line_y, color="red", linewidth=2, label="Regression Line")
plt.xlabel("Height (inches)")
plt.ylabel("Weight (pounds)")
plt.title("Height vs Weight with Regression Line")
plt.legend()
plt.grid(True)
plt.show()


# --------------------------------------------
# STEP 9: Residual Plot
# --------------------------------------------
residuals = y_test - y_test_pred

plt.figure(figsize=(8, 5))
plt.scatter(y_test_pred, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Weight")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.grid(True)
plt.show()
