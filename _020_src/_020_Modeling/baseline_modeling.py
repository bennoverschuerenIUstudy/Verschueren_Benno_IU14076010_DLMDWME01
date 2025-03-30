import pandas as pd
import time

from darts import TimeSeries
from darts.models import NaiveSeasonal
from darts.models import Prophet
from darts.models import Theta
from darts.models import NBEATSModel
from darts.models import ExponentialSmoothing
from darts.models import XGBModel
from darts.metrics import mape

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import _020_src._global_parameters as gp
from _020_src._010_DataPrep import data_prep as dp
from _020_src._030_Deployment import custom_functions as cf



df = dp.load_prepare_data(PATH="_010_data/_010_Raw/sickness_table.csv")

series = TimeSeries.from_dataframe(df, time_col=None, value_cols="calls")
#series = TimeSeries.from_dataframe(df, time_col=None, value_cols="n_sick")

train, test = series.split_after(0.8)

# Baselinemodels

#model = NaiveSeasonal(K=365)
#model_fc = NaiveSeasonal(K=365)

#model = Theta(seasonality_period=365)
#model_fc = Theta(seasonality_period=365)

#model = ExponentialSmoothing(seasonal_periods=365)
#model_fc = ExponentialSmoothing(seasonal_periods=365)

#model = NBEATSModel(input_chunk_length=365, output_chunk_length=1)
#model_fc = NBEATSModel(input_chunk_length=365, output_chunk_length=1)

model = Prophet()
model_fc = Prophet()

# Fit to traindata
start_time_train = time.time()
model.fit(train)
# Predict testdata
start_time_predict = time.time()
predict = model.predict(len(test))
# Forecast (out-of-sample)
days = 365
start_time_forecast = time.time()
model_fc.fit(series)
forecast = model_fc.predict(days)


# Evaluate
print(f"Model: {model.__class__.__name__}")
# Get time_train
train_time = time.time() - start_time_train
predict_time = time.time() - start_time_predict
forecast_time = time.time() - start_time_forecast
print(f"Time_train / predict / forecast: {train_time*1000:.3f}ms / {predict_time*1000:.3f}ms / {forecast_time*1000:.3f}ms")

# Get MAPE
error = mape(test, predict)
print(f"MAPE: {error:.2f}%")

# Plot data
fig, ax = plt.subplots(figsize=(16, 4))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

# Traindata
train.plot(label="Train Data", color='black', linewidth=1, ax=ax)

# Testdata
test.plot(label="Test Data", color='0.75', linewidth=1, ax=ax)

# Prediction
predict.plot(label="Prediction", color=gp._COLOR_, linewidth=1, ax=ax)

# Forecast
forecast.plot(label="Forecast", color=gp._COLOR_, linestyle=":", linewidth=1, ax=ax)

# Setup plot
ax.legend(loc="best", fontsize=gp._FONTSIZE_LEGEND_)
ax.tick_params(labelsize=gp._FONTSIZE_TICK_)

plt.tight_layout(pad=2.0)

plt.show()