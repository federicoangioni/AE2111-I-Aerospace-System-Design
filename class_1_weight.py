import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Initial data
payload = [8190, 9840, 8976, 8190, 8629, 10478, 8935, 13047, 12245, 8554]
MTOW = [32999, 38600, 42800, 34000, 29480, 44000, 40500, 50300, 49450, 35990]

# Reshape data for the regression model
X = np.array(payload).reshape(-1, 1)
y = np.array(MTOW)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Print the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")

# Predict MTOW values based on the model
y_pred = model.predict(X)