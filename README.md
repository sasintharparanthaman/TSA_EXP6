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
<img width="741" height="538" alt="Screenshot 2026-03-16 094738" src="https://github.com/user-attachments/assets/b6af4974-60bb-430b-8cfe-8905eef9d2be" />

<img width="763" height="549" alt="Screenshot 2026-03-16 094753" src="https://github.com/user-attachments/assets/fa2c894b-e7f3-4d7d-847e-50da730cc470" />

<img width="792" height="567" alt="Screenshot 2026-03-16 094818" src="https://github.com/user-attachments/assets/b5a5ac42-4f29-4330-b5ad-c92f55af9d58" />

<img width="765" height="527" alt="Screenshot 2026-03-16 094833" src="https://github.com/user-attachments/assets/0d95852d-7898-42a9-b192-7a77b3a9dbf1" />

<img width="280" height="89" alt="Screenshot 2026-03-16 094840" src="https://github.com/user-attachments/assets/5ab0a6ca-1217-4962-a520-83fdab95ac08" />


<img width="782" height="546" alt="Screenshot 2026-03-16 094904" src="https://github.com/user-attachments/assets/44757fe7-cd6b-4d8b-ba9f-90cafb162fd7" />


### RESULT:

Thus the program run successfully based on the Holt Winters Method model.
