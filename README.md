# Ex.No: 6               HOLT WINTERS METHOD

### AIM:

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
data = pd.read_csv('/content/bike_sales_india.csv')

# Convert Registration Year to datetime
data['Registration Year'] = pd.to_datetime(data['Registration Year'], format='%Y')

# Set Registration Year as index
data.set_index('Registration Year', inplace=True)

# Select Resale Price column for time series
data_yearly = data['Resale Price (INR)'].resample('Y').mean()

print(data_yearly.head())

# Plot original data
data_yearly.plot()
plt.title("Yearly Resale Price")
plt.show()

# Scaling
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_yearly.values.reshape(-1,1)).flatten(),
    index=data_yearly.index
)

scaled_data.plot()
plt.title("Scaled Data")
plt.show()

# Decomposition
decomposition = seasonal_decompose(data_yearly, model="additive")
decomposition.plot()
plt.show()

# Multiplicative seasonality cannot handle non-positive values
scaled_data = scaled_data + 1

# Train Test Split
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

# Holt-Winters Model
model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=2
).fit()

# Forecast
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot evaluation
ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)

ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual evaluation')
plt.show()

# RMSE
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("RMSE:", rmse)

# Standard deviation and mean
print("Std Dev:", np.sqrt(scaled_data.var()))
print("Mean:", scaled_data.mean())

# Final Model
final_model = ExponentialSmoothing(
    scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=2
).fit()

# Predict future resale price
final_predictions = final_model.forecast(steps=3)

# Plot prediction
ax = scaled_data.plot()
final_predictions.plot(ax=ax)

ax.legend(["Actual Data", "Future Predictions"])
ax.set_xlabel('Year')
ax.set_ylabel('Resale Price')
ax.set_title('Resale Price Prediction')

plt.show()
```
### OUTPUT:




### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
