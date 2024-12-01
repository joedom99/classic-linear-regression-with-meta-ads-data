# Linear Regression: The Classic Machine Learning Algorithm You Need to Know
# Example Python script to perform linear regression on Meta (Facebook) Ads data
# By: Joe Domaleski

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load the Meta Ads data
data = pd.read_csv("meta_ads_data_summer_campaign.csv")

# Perform linear regression: clicks as dependent variable, impressions as independent variable
X = data['impressions']  # Independent variable
y = data['clicks']       # Dependent variable

# Add a constant to the independent variable for the intercept term
X_with_const = sm.add_constant(X)

# Fit the regression model
model = sm.OLS(y, X_with_const).fit()

# Display the summary of the regression model
print(model.summary())

# Create a scatter plot with regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', alpha=0.6, label="Observed Data")
plt.plot(X, model.predict(X_with_const), color='red', label="Regression Line")
plt.title("Regression Analysis: Clicks vs Impressions")
plt.xlabel("Impressions")
plt.ylabel("Clicks")
plt.legend()
plt.grid(True)
plt.show()

# Create a residual plot to check model diagnostics
residuals = y - model.predict(X_with_const)
plt.figure(figsize=(10, 6))
plt.scatter(model.predict(X_with_const), residuals, color='blue', alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--', linewidth=1)
plt.title("Residual Plot")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.grid(True)
plt.show()

# Predict clicks for new impressions
# This demonstrates how to use the model to make predictions
new_data = pd.DataFrame({'impressions': [100000, 200000, 300000]})
new_data_with_const = sm.add_constant(new_data)
predictions = model.predict(new_data_with_const)

# Print the predictions without scientific notation
pd.options.display.float_format = '{:.0f}'.format  # Disable scientific notation
print(pd.DataFrame({'Impressions': new_data['impressions'], 'Predicted Clicks': predictions.round()}))

