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
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.dates as mdates

def Fit_Predict_Prophet(df, col):
    """ Erstellt das Prophet-Modell, trainiert es und evaluiert die Genauigkeit. 
        Gibt das trainierte Modell zurück und zeigt den Train-Test-Split-Plot. """

    # Daten für Prophet vorbereiten
    df_prophet = df[[col]].reset_index()
    df_prophet.rename(columns={'date': 'ds', col: 'y'}, inplace=True)

    # Train-Test-Split (80% Training, 20% Test)
    train_size = int(len(df_prophet) * 0.8)
    train, test = df_prophet[:train_size], df_prophet[train_size:]

    # Modell erstellen und trainieren
#    model = Prophet(changepoint_prior_scale=0.1, seasonality_prior_scale=10)
#    model.add_seasonality(name='yearly', period=365, fourier_order=400)  # Jährliche Saisonalität

    model = Prophet(changepoint_prior_scale=0.1, seasonality_prior_scale=10)
    model.add_seasonality(name='yearly', period=365, fourier_order=100)  # Jährliche Saisonalität
    model.add_seasonality(name='monthly', period=30, fourier_order=10)  # Jährliche Saisonalität

    model.fit(train)

    # Test-Vorhersagen
    test_future = test[['ds']]
    test_forecast = model.predict(test_future)

    # RMSE berechnen
    rmse = np.sqrt(mean_squared_error(test['y'], test_forecast['yhat']))
    percentage_rmse = (rmse / np.mean(test['y'])) * 100
    print(f'Prozentualer RMSE: {percentage_rmse:.2f}%')

    # MAE berechnen
    mae = mean_absolute_error(test['y'], test_forecast['yhat'])
    percentage_mae = (mae / np.mean(test['y'])) * 100
    print(f'Prozentualer MAE: {percentage_mae:.2f}%')

    # Visualisierung: Train-Test-Split
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.plot(df_prophet['ds'], df_prophet['y'], label='Train data', color='black', linewidth=1)
    ax.plot(test['ds'], test['y'], label='Test data', color='0.75', linestyle='-', linewidth=1)
    ax.plot(test_forecast['ds'], test_forecast['yhat'], label='Prediction', color='red', linestyle='-', linewidth=1)
    ax.axvline(x=test['ds'].min(), color='black', linestyle='--', label='Train-Test-Split')

    ax.set_ylabel("Calls")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.show()

    return model


def Forecast(model, period):
    """ Erstellt eine Forecast für die angegebene Periodendauer. 
        Zeigt den Forecast-Plot und gibt die Vorhersage-Daten zurück. """

    # Zukünftige Daten für die Prognose
    last_date = model.history['ds'].max()
    future_dates = pd.DataFrame({'ds': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=period, freq='D')})
    forecast = model.predict(future_dates)

    # Visualisierung des Forecasts
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.plot(model.history['ds'], model.history['y'], label='Historical Data', color='black', linewidth=1)
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red', linestyle='-', linewidth=1)

    ax.set_ylabel("Calls")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.show()

    return forecast


model = Fit_Predict_Prophet(df, "n_sick_modified")
#forecast_df = Forecast(model, period=90)