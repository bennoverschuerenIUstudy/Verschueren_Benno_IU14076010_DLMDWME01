from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# Laden der Daten
df = pd.read_csv("_data/sickness_table.csv", parse_dates=True)

# Die ersten paar Zeilen der Rohdaten anzeigen
print("\n\nSHOW RAW DATA\n----------------------------------------------------------------------------")
print(df.head())  # Zeigt die ersten Zeilen des DataFrames
print("----------------------------------------------------------------------------\n")

# Setze die 'date' Spalte als Index und konvertiere sie zu datetime
df.set_index('date', inplace=True)
df.index = pd.to_datetime(df.index)
df = df.asfreq('D')  # 'D' für tägliche Daten

# Setze den Datentyp der 'calls' Spalte auf integer
df['calls'] = df['calls'].astype(int)

# Entferne unerwünschte Spalten (wie z.B. 'Unnamed')
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]


from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates

# Annahme: df enthält die Daten mit 'date' als Index und 'calls' als Zielvariable

# Bereite das DataFrame für Prophet vor
df_prophet = df[['calls']].reset_index()
df_prophet.rename(columns={'date': 'ds', 'calls': 'y'}, inplace=True)

# Split in Trainings- und Testdaten (z.B. 80% Training, 20% Test)
train_size = int(len(df_prophet) * 0.8)
train, test = df_prophet[:train_size], df_prophet[train_size:]

#train["y"]*=1.14


model = Prophet(changepoint_prior_scale=0.1, seasonality_prior_scale=10)
model.add_seasonality(name='yearly', period=365, fourier_order=400)  # Höhere Fourier-Ordnung für jährliche Saisonalität

#model.add_seasonality(name='weekly_on_season', period=365, fourier_order=400)
#model.add_seasonality(name='weekly_off_season', period=7, fourier_order=3)



model.fit(train)


# Testvorhersagen treffen
test_future = test[['ds']]
test_forecast = model.predict(test_future)

#model.plot_components(test_forecast)

# Berechnung des prozentualen RMSE
rmse = np.sqrt(mean_squared_error(test['y'], test_forecast['yhat']))
percentage_rmse = (rmse / np.mean(test['y'])) * 100
print(f'Prozentualer RMSE: {percentage_rmse:.2f}%')

# Berechnung des prozentualen MAE
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test['y'], test_forecast['yhat'])
percentage_mae = (mae / np.mean(test['y'])) * 100
print(f'Prozentualer MAE: {percentage_mae:.2f}%')


# Forecast für das nächste Monat (30 Tage), beginnend nach den Testdaten
last_test_date = test['ds'].max()
future_month = pd.DataFrame({'ds': pd.date_range(start=last_test_date + pd.Timedelta(days=1), periods=90, freq='D')})
forecast_month = model.predict(future_month)




# Visualisierung


# Generate date range for x-axis ticks
major_locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
formatter = mdates.DateFormatter('%Y-%m-%d')
fig, ax = plt.subplots(figsize=(16,4))

ax.xaxis.set_major_locator(major_locator)
ax.xaxis.set_major_formatter(formatter)


plt.plot(df_prophet['ds'], df_prophet['y'], label='Train data', color='black', linewidth=1)
plt.plot(test['ds'], test['y'], label='Test data', color='0.75', linestyle='-', linewidth=1)
plt.plot(test_forecast['ds'], test_forecast['yhat'], label='Prediction', color='red', linestyle='-', linewidth=1)
plt.plot(forecast_month['ds'], forecast_month['yhat'], label='Forecast', color='green', linestyle='-', linewidth=1)
plt.axvline(x=test['ds'].min(), color='black', linestyle='--', label='Train-Test-Split')
plt.ylabel("calls")
ax.legend(loc="best", fontsize=8)

ax.tick_params(labelsize=8)

# Add gridlines
ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

# Modify plot frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show plot
plt.tight_layout(pad=2.0)
plt.show()
