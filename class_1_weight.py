import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("reference.csv")

# Initial data
oew = df["OEM [ton]"]
MTOW_ref = [32999, 38600, 42800, 34000, 29480, 44000, 40500, 50300, 49450, 35990]

# Reshape data for the regression model
X = np.array(oew).reshape(-1, 1)
y = np.array(MTOW_ref)

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Print the coefficients
print(f"Intercept: {model.intercept_}")
print(f"Slope: {model.coef_[0]}")

# Predict MTOW values based on the model
def predict(self):
    mtow = model.predict(self)
    return mtow