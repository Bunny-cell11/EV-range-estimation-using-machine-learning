import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# Suppress the UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Data
data = {
    'battery_capacity': [50, 60, 70, 55, 65],#capacity of battery in kilowatts/kwh
    'vehicle_weight': [1500, 1600, 1550, 1580, 1620],# Vehicle weight in kg(kilograms)
    'driving_conditions': [1, 2, 1, 3, 2],# Driving conditions (categorical i.e;1=Urban region,2=highway region,3=mixed/hilly region)
    'temperature': [20, 25, 15, 30, 22], # Temperature in degrees Celsius °C
    'range': [300, 350, 370, 320, 360]# Range in kilometers (km)
}


df = pd.DataFrame(data)

# Save DataFrame to CSV
df.to_csv('ev_data.csv', index=False)

print("Sample 'ev_data.csv' file created.")

# Load dataset
data = pd.read_csv('ev_data.csv')

# Display the first few rows
print(data.head())

# Features and target variable
X = df[['battery_capacity', 'vehicle_weight', 'driving_conditions', 'temperature']]
y = df['range']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Polynomial features for non-linear regression
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Models
lin_reg = LinearRegression()
ridge_reg = Ridge(alpha=1.0)
lasso_reg = Lasso(alpha=0.1)

# Train models on the entire dataset
lin_reg.fit(X_poly, y)
ridge_reg.fit(X_poly, y)
lasso_reg.fit(X_poly, y)

# Predictions on the entire dataset
y_pred_lin = lin_reg.predict(X_poly)
y_pred_ridge = ridge_reg.predict(X_poly)
y_pred_lasso = lasso_reg.predict(X_poly)

# Evaluation function
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, r2

# Linear Regression Evaluation
mae_lin, mse_lin, r2_lin = evaluate_model(y, y_pred_lin)

# Ridge Regression Evaluation
mae_ridge, mse_ridge, r2_ridge = evaluate_model(y, y_pred_ridge)

# Lasso Regression Evaluation
mae_lasso, mse_lasso, r2_lasso = evaluate_model(y, y_pred_lasso)

print(f"Linear Regression - MAE: {mae_lin}, MSE: {mse_lin}, R²: {r2_lin}")
print(f"Ridge Regression - MAE: {mae_ridge}, MSE: {mse_ridge}, R²: {r2_ridge}")
print(f"Lasso Regression - MAE: {mae_lasso}, MSE: {mse_lasso}, R²: {r2_lasso}")

print("Actual values: ", y.tolist())
print("Predicted values (Linear): ", y_pred_lin.tolist())
print("Predicted values (Ridge): ", y_pred_ridge.tolist())
print("Predicted values (Lasso): ", y_pred_lasso.tolist())

# Visualization
plt.figure(figsize=(12, 6))

# Linear Regression
plt.subplot(1, 3, 1)
plt.scatter(y, y_pred_lin, alpha=0.5)
plt.title('Linear Regression')
plt.xlabel('Actual Range')
plt.ylabel('Predicted Range')

# Ridge Regression
plt.subplot(1, 3, 2)
plt.scatter(y, y_pred_ridge, alpha=0.5)
plt.title('Ridge Regression')
plt.xlabel('Actual Range')
plt.ylabel('Predicted Range')

# Lasso Regression
plt.subplot(1, 3, 3)
plt.scatter(y, y_pred_lasso, alpha=0.5)
plt.title('Lasso Regression')
plt.xlabel('Actual Range')
plt.ylabel('Predicted Range')

plt.tight_layout()
plt.show()# EV-range-estimation-using-machine-learning
