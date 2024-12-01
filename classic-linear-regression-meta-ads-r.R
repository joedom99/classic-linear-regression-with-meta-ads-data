# Linear Regression: The Classic Machine Learning Algorithm You Need to Know
# Example R script to do linear regression on Meta (Facebook) Ads
# By: Joe Domaleski

# Load necessary libraries
library(ggplot2)

# Load the Meta Ads data
data <- read.csv("meta_ads_data_summer_campaign.csv")

# Perform linear regression: clicks as dependent variable, impressions as independent variable
lm_model <- lm(clicks ~ impressions, data = data)

# Display the summary of the regression model
# This shows coefficients, R-squared, and significance levels
summary(lm_model)

# Create a scatter plot with regression line
ggplot(data, aes(x = impressions, y = clicks)) +
  geom_point(color = "blue", alpha = 0.6) +  # Scatter plot points
  geom_smooth(method = "lm", color = "red", se = TRUE) +  # Regression line
  labs(
    title = "Regression Analysis: Clicks vs Impressions",
    x = "Impressions",
    y = "Clicks"
  ) +
  theme_minimal()

# Create a residual plot to check model diagnostics
residuals <- resid(lm_model)
fitted_values <- fitted(lm_model)
ggplot(data.frame(Fitted = fitted_values, Residuals = residuals), aes(x = Fitted, y = Residuals)) +
  geom_point(color = "blue", alpha = 0.6) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(
    title = "Residual Plot",
    x = "Fitted Values",
    y = "Residuals"
  ) +
  theme_minimal()

# Predict clicks for new impressions
# This demonstrates how to use the model to make predictions
new_data <- data.frame(impressions = c(100000, 200000, 300000))
predictions <- round(predict(lm_model, new_data))  # Round predicted clicks for readability
options(scipen = 999)  # Disable scientific notation
print(data.frame(new_data, Predicted_Clicks = predictions))

