import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import custom_functions as cf


from statsmodels.tsa.stattools import adfuller

# Load data and set column "date" as index
df = pd.read_csv("_data/sickness_table.csv", parse_dates=True)

# Print data
print("\n\nSHOW RAW DATA\n----------------------------------------------------------------------------")
print(df.head())  # Show the DataFrame
print("----------------------------------------------------------------------------\n")


# Convert to datetime
df.set_index('date', inplace=True)
df.index = pd.to_datetime(df.index)
df = df.asfreq('D')  # 'D' steht für Daily
# Set the df to integer
df = df.astype(int)
# Delete unnamed column
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
print("\n\nSHOW MODIFIED DATA\n-------------------------------------------------------------")
print(df.head())  # Show the DataFrame
print("-------------------------------------------------------------\n")


cf.check_data(df)

# Plot entire data
#cf.plot_data(df, 
#             "2016-04-01", 
#             "2019-08-31", 
#             "Q" )

# Plot outliers
#cf.plot_outliers(df[["n_sick"]], 
#                 cf.detect_outliers_iqr, 
#                 "2016-04-01", "2019-08-31", 
#                 (16,4))


# Interpolate outlier
df["n_sick_modified"] = df["n_sick"]
df.at["2017-10-29", 'n_sick_modified'] = cf.linear_interpolation(df["n_sick_modified"]["2017-10-28"], df["n_sick_modified"]["2017-10-30"])

#cf.plot_data(df[["n_sick_modified"]], 
#             "2016-04-01", "2019-08-31", 
#             "Q", 
#             (16,4))


# Show reference from accidents statistics
#cf.show_referenceDataAccidents()
#cf.show_referencePopulation()


# Add n_sby to n_duty, because it is 24/7/365 booked
df["n_duty_real"] = df["n_duty"] + df["n_sby"]
# Offset n_duty with n_sick
df["n_duty_real"] -= df["n_sick"]

#cf.plot_combined_datasets(df[["n_duty_real"]], 
#                          df[["n_duty"]], 
#                          "2016-04-01", "2019-08-31", 
#                          "Q", 
#                          (16,4))


# Initialisiere Variablen, um den besten Faktor und die höchste Korrelation zu speichern
max_corr_factor = None
highest_correlation = float('-inf')  # Kleinster möglicher Wert als Startpunkt

for i in np.arange(4, 6, 0.001):
    # Berechne die modifizierten Anrufe
    df["calls_by_sby"] = df["calls"] - df["n_duty_real"] * i
    df.loc[df["calls_by_sby"] < 0, "calls_by_sby"] = 0

    # Berechne die Korrelation
    correlation = cf.get_correlation(df["calls_by_sby"], 
                                     df["dafted"])

    # Überprüfe, ob die aktuelle Korrelation die höchste ist
    if correlation > highest_correlation:
        highest_correlation = correlation
        max_corr_factor = i


# Ausgabe des besten Faktors und der höchsten Korrelation
print("\n\nGET MAX CORRELATION\n-------------------------------")
print(f"MAX_CORR_FACTOR:\t{max_corr_factor:.3f}\nCORRELATION:\t\t{highest_correlation:.3f}")
print("-------------------------------\n")


# Offset the calls with the amount of workers
df["calls_by_sby"] = df["calls"] - df["n_duty_real"] * max_corr_factor
df["calls_by_duty"] = df["calls"] - df["n_duty_real"] * max_corr_factor

# Cut the values under/over 0
df.loc[df["calls_by_sby"] < 0, "calls_by_sby"] = 0
df.loc[df["calls_by_duty"] > 0, "calls_by_duty"] = 0

# Calls_by_sby must also be offsetted by the max_corr_factor
df["calls_by_sby"] = (df["calls_by_sby"] / max_corr_factor).round().astype(int)
df["calls_by_duty"] = (df["calls_by_duty"] / max_corr_factor).round().astype(int)


# Compare sby_need with real calls_by_sby 
#cf.plot_data(df[["calls_by_sby", "dafted"]],
#            "2016-04-01", "2019-08-31", 
#             "Q",
#             (16,8))


# Plot correlation of "calls" and "dafted"
#cf.scatter_correlation(df["dafted"], 
#                       df["calls_by_sby"])


# Add calls to n_duty_real => how many drivers we really need (based on calls)?
df["n_duty_required"] = df["n_duty_real"] + df["calls_by_sby"]
df["n_duty_required"] = df["n_duty_required"] + df["calls_by_duty"]

#cf.plot_data(df[["calls", "calls_by_sby", "calls_by_duty", "n_duty_required"]], 
#             "2016-04-01", "2019-08-31", 
#             "Q")


# Get correlation of the calculated required duty and calls
#print(cf.scatter_correlation(df["n_duty_required"], df["calls"]))


# Show the difference of n_duty and the required duty
#cf.plot_combined_datasets(df[["n_duty_required"]], df[["n_duty"]], 
#                          "2016-04-01", "2019-08-31", 
#                          "Q",
#                          FIGSIZE=(16,4))


# Get the active drivers (based on the calls) and duty_real
df["active_drivers"] = df["n_duty_real"] + df["dafted"]


#cf.plot_data(df[["active_drivers"]], 
#            "2016-04-01", "2019-08-31", 
#            "Q",
#            (16, 4))

# Compare planned duty and needed duty (based on calls) and plot bargraph (Yearly)
#cf.plot_metrics_comparison(df[["n_duty_required"]], 
#                           df[["active_drivers"]])


# -------------------
# Prediction_BASELINE
# -------------------
"""# Create Dataframe for predictions
df_pred_baseline = cf.baseline_prediction(df,
                                        FREQ="Q")

# Save calls_baseline_prediction
df_pred_baseline.rename(columns={"calls": "calls_baseline"}, inplace=True)
# Compare prediction with reality
cf.plot_combined_datasets(df_pred_baseline,
                        df[["calls"]],
                        "2018-10-09", "2019-05-27",
                        "M",
                        COL_DF2="0.65",
                        LINESTYLE_DF2="-",
                        LINEWIDTH_DF1=3,
                        FIGSIZE=(16, 8))
"""

"""# Create Dataframe for calculated n_duties (based on prediction and max_corr_factor)
df_duty_predicted = (df_pred_baseline / max_corr_factor).astype(int)
df_duty_predicted.rename(columns={"calls_baseline": "n_duty_pred_baseline"}, inplace=True)

# Compare prediction with reality
cf.plot_combined_datasets(df_duty_predicted[["n_duty_pred_baseline"]],
                          df[["n_duty_required"]],
                        "2018-10-09", "2019-05-27",
                        "M",
                        COL_DF2="0.65",
                        LINESTYLE_DF2="-",
                        LINEWIDTH_DF1=3,
                        FIGSIZE=(16, 8))"""



# Create dataframe for prediction (copy of df)
df_pred = df[['calls']].copy()
df_pred.index = df.index  # Index übernehmen




# -----------------
# Prediction_SARIMA
# -----------------

# 1. Check the stationarity of the calls for further forecasting
#cf.check_stationarity(df[["calls"]])

# Decomp data to see trend and seasonality
#decomp = cf.plot_seasonal_decomposition(df, "calls")

# Find values via acf/pacf plot (https://arauto.readthedocs.io/en/latest/how_to_choose_terms.html)
#cf.plot_acf_pacf(decomp.seasonal.diff().dropna()) # (9,1,1)(1,1,2)(0)



"""
# Train-Test-Split (80% Training, 20% Test)
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# SARIMA
# LONG PROGRESSTIME !!!
pred_SARIMA = cf.SARIMA(df, "calls", p=3, d=1, q=1, P=1, D=1, Q=2, TRAIN=train, TEST=test, S=365)
df_pred_SARIMA = pd.DataFrame(pred_SARIMA, index=test.index)

cf.plot_combined_datasets(df_pred_SARIMA,
                        df[["calls"]],
                        "2018-10-09", "2019-05-27",
                        "M",
                        COL_DF2="0.65",
                        LINESTYLE_DF2="-",
                        LINEWIDTH_DF1=3,
                        FIGSIZE=(16, 8))


cf.scatter_correlation(test["calls"], pred_SARIMA, X_LABEL="Prediction", Y_LABEL="Value")
cf.validate(test["calls"], pred_SARIMA)

"""

#"2016-04-01", "2019-08-31"




# HYBRID SARIMA
# -----------------

# Resample data to weekly -> Get Seasonality via SARIMA
# Interpolate back to daily
# Get daily residuals via ARIMA
# Add them


# Resample daily to weekly data
df_calls_weekly = df_pred.resample('W-FRI').mean().asfreq('W-FRI')

# Interpolate weekly back to daily for further operations
df_pred["calls_weekly"] = df_calls_weekly.resample("D").interpolate(method="linear") # Werte interpolieren
df_pred["calls_diff"] = df_pred["calls"] - df_pred["calls_weekly"]

# Show calls (daily) and calls (weekly)
cf.plot_combined_datasets(df_pred[["calls_weekly"]], 
                          df_pred[["calls"]], 
                          "2016-04-01", "2019-08-31", 
                          LINESTYLE_DF2="-", 
                          COL_DF2="0.75", 
                          FREQ="Q", 
                          LINEWIDTH_DF1=2)



"""# Find ARIMA parameters
from pmdarima import auto_arima
# Automatische Suche nach den besten SARIMA-Parametern
model = auto_arima(
    df_weekly["calls"],               # Zielvariable
    seasonal=True,             # Berücksichtigt Saisonalität
    m=52,                      # Saisonalitätsperiode (z. B. 12 für monatliche Daten)
    stepwise=True,             # Beschleunigt die Berechnung
    suppress_warnings=True,    # Unterdrückt Warnungen
    trace=True                 # Zeigt den Fortschritt der Modellwahl
)

# Beste Parameter anzeigen
print(model.summary())  # Best model:  ARIMA(2,1,4)(0,0,2)[52]
2 1 4 / 1 0 0

"""

# Decomp data to see trend and seasonality
#decomp = cf.plot_seasonal_decomposition(df_weekly, "calls", PERIOD=52)
# Get value for the lag
#cf.plot_acf_pacf(decomp.seasonal.diff().dropna(), LAGS=60)

# Train-Test-Split for weekly prediction
train_size = int(len(df_calls_weekly) * 0.8)
train_weekly, test_weekly = df_calls_weekly[:train_size], df_calls_weekly[train_size:]

# Best model:  ARIMA(2,1,4)(0,0,2)[52]
# Create model based on weekly data
prediction_W_SARIMA = cf.SARIMA(df_calls_weekly, "calls", train_weekly, test_weekly, p=2, d=1, q=4, P=1, D=1, Q=0, S=52)
df_pred_W_SARIMA = pd.DataFrame(prediction_W_SARIMA, index=test_weekly.index)
df_pred_W_SARIMA.rename(columns={"predicted_mean": "calls_predicted"}, inplace=True)

# Resample weekly data to daily
df_pred["SARIMA_weekly"] = df_pred_W_SARIMA.resample("D").interpolate(method="linear") # Werte interpolieren

cf.plot_combined_datasets(df_pred_W_SARIMA,
                        df_calls_weekly[["calls"]],
                        "2018-10-09", "2019-05-27",
                        "M",
                        COL_DF2="0.65",
                        LINESTYLE_DF2="-",
                        LINEWIDTH_DF1=3,
                        FIGSIZE=(16, 8))

# Check model
cf.scatter_correlation(test_weekly["calls"], prediction_W_SARIMA, X_LABEL="Prediction", Y_LABEL="Value")
cf.validate(test_weekly["calls"], prediction_W_SARIMA)



# ------------------
# # Prediction_XGBoost -> YEAHHHH!
# ------------------
import xgboost as xgb
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error


# 1. Get Features erstellen: Historische Werte + Zeitmerkmale
df_pred['day_of_year'] = df_pred.index.dayofyear
df_pred['week_of_year'] = df_pred.index.isocalendar().week
df_pred['day_of_week'] = df_pred.index.dayofweek

# Lag-Features (Vergangene Werte als Input)
lags = [1, 7, 30, 365]  # 1 Tag, 1 Woche, 1 Monat, 1 Jahr
for lag in lags:
    df_pred[f'lag_{lag}'] = df_pred['calls'].shift(lag)

#df_pred["calls"] = df_pred["calls"].diff()

# Fehlende Werte durch Shifts entfernen
df_pred.dropna(inplace=True)


# 2️⃣ Daten splitten: Training & Test
split_ratio = 0.8
split_index = int(len(df_pred) * split_ratio)
train, test = df_pred.iloc[:split_index], df_pred.iloc[split_index:]

# Features und Zielvariable definieren
X_train = train.drop(columns=["calls", "calls_weekly", "calls_diff", "SARIMA_weekly"])
y_train = train['calls_diff']
#y_train = train['calls']
X_test = test.drop(columns=["calls", "calls_weekly", "calls_diff", "SARIMA_weekly"])
y_test = test['calls_diff']
#y_test = test['calls']

# 3️⃣ XGBoost Modell trainieren
model = xgb.XGBRegressor(n_estimators=2000, learning_rate=0.1, objective='reg:squarederror')
model.fit(X_train, y_train)

# 4️⃣ Vorhersage für den Testzeitraum
pred_xgb = model.predict(X_test)
df_pred["XGBoost_daily"] = pd.DataFrame(pred_xgb, columns=['calls'], index=test.index)
df_pred["prediction_final"] = df_pred["SARIMA_weekly"] + df_pred["XGBoost_daily"]

# Erstelle das Plot
major_locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
formatter = mdates.DateFormatter('%Y-%m-%d')

fig, ax = plt.subplots(figsize=(16, 4))

ax.plot(train.index, train["calls"], label="Trainingdata", color="black")
ax.plot(test.index, test["calls"], label="Testdata", color="0.75")
ax.plot(df_pred.index, df_pred["prediction_final"], label="Prediction", color="red")
    
# Setze Labels und Legende
#ax.set_ylabel(COLUMN)    
ax.legend(loc="best", fontsize=8)

# Setze Major-Ticks und Formatierung
ax.xaxis.set_major_locator(major_locator)
ax.xaxis.set_major_formatter(formatter)
ax.tick_params(labelsize=6)

# Füge Gitterlinien hinzu
ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

# Modifiziere den Rahmen des Plots
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Zeige den Plot
plt.tight_layout()
plt.show()

cf.plot_combined_datasets(df_pred[["prediction_final"]],
                        test[["calls"]],
                        "2018-10-09", "2019-05-27",
                        "M",
                        COL_DF2="0.65",
                        LINESTYLE_DF2="-",
                        LINEWIDTH_DF1=3,
                        FIGSIZE=(16, 8))


# 7️⃣ Validierung (falls du cf.validate nutzen möchtest)
#df_forecast = pd.DataFrame(df_pred, index=df_pred.index, columns=["prediction_final"])

#pred_final = pred_xgb[""] + prediction_W_SARIMA
temp = df_pred["prediction_final"].iloc[split_index:]
print(temp)
print(test["calls"])
cf.scatter_correlation(test["calls"], df_pred["prediction_final"].iloc[split_index:], X_LABEL="Prediction", Y_LABEL="Value")
cf.validate(test["calls"], df_pred["prediction_final"].iloc[split_index:])

# Check model
#cf.scatter_correlation(test_weekly["calls"], prediction_W_SARIMA, X_LABEL="Prediction", Y_LABEL="Value")
#cf.validate(test_weekly["calls"], prediction_W_SARIMA)















# Calculate residuals
# -------------------
# 1. Resample daily data to weekly data
# 2. Resample back
# 3. df - df_sampled = residuals
# 4. train model for residuals




# 3. Calculate Residuals
#df_residuals = (df["calls"] - df_daily_resampled["calls"]).dropna().to_frame()
#df_residuals.columns = ["calls"]
#print(df_residuals)

#df_pred["calls_diff"] = (df_pred["calls"] - df_pred["df_pred_W_SARIMA"]).dropna().to_frame()
#df_residuals.columns = ["calls"]
#print(df_residuals)



#cf.plot_seasonal_decomposition(df_residuals, "calls")










# Prediction of Residuals via ExponentialSmoothin -> not good
"""from statsmodels.tsa.holtwinters import ExponentialSmoothing

split_ratio = 0.8
split_index = int(len(df) * split_ratio)
train, test = df_residuals.iloc[:split_index], df_residuals.iloc[split_index:]

# 📌 Holt-Winters-Modell (Additiv für stabile saisonale Schwankungen)
model = ExponentialSmoothing(train['calls'], trend='add', seasonal='add', seasonal_periods=365)
model_fit = model.fit()

# 📌 Vorhersage für den Testzeitraum
forecast = model_fit.forecast(steps=len(test))

# 📌 Visualisierung der Ergebnisse
plt.figure(figsize=(12, 6))
plt.plot(df_residuals.index, df_residuals['calls'], label='Original Data', alpha=0.6)
plt.plot(test.index, test['calls'], label='Test Data', color='green')
plt.plot(test.index, forecast, label='Holt-Winters Forecast', color='red', linestyle='dashed')
plt.legend()
plt.title('Holt-Winters Modell für tägliche Zeitreihe mit jährlicher Saisonalität')
plt.show()


# Check model
df_forecast = pd.DataFrame(forecast, index=test.index)
df_forecast.rename(columns={0: "calls_predicted"}, inplace=True)
cf.scatter_correlation(df_forecast["calls_predicted"], test["calls"], X_LABEL="Prediction", Y_LABEL="Value")
cf.validate(df_forecast["calls_predicted"], test["calls"])"""





# Prediction of Residuals via ARIMA -> not good
# 4a. Get parameters for arima model (residuals)
"""from pmdarima import auto_arima
model = auto_arima(
    df_residuals["calls"],               # Zielvariable
    seasonal=False,             # Berücksichtigt Saisonalität
    #m=52,                      # Saisonalitätsperiode (z. B. 12 für monatliche Daten)
    stepwise=True,             # Beschleunigt die Berechnung
    suppress_warnings=True,    # Unterdrückt Warnungen
    trace=True                 # Zeigt den Fortschritt der Modellwahl
)

# Beste Parameter anzeigen
print(model.summary())  # Best model:  ARIMA(2,0,2)(0,0,0)[0]

"""

"""# 4b. Train model 
train_noise_size = int(len(df_residuals) * 0.8)
train_noise, test_noise = df_residuals[:train_noise_size], df_residuals[train_noise_size:]

#print(train_noise)
print(test_noise)
#print(df_residuals)


from statsmodels.tsa.arima.model import ARIMA

# ARMA-Modell auf die Residuen anwenden (AR=1, MA=1 Beispiel)
model_residuals = ARIMA(train_noise["calls"], order=(40, 0, 30))  # ARMA(1, 0, 1)
fitted_model_residuals = model_residuals.fit()

# Vorhersage der Residuen (Rauschen)
prediction_noise = fitted_model_residuals.forecast(steps=len(test_noise))

# Erstelle DataFrame mit Vorhersage und Index von test_noise
df_prediction_noise = pd.DataFrame(prediction_noise, index=test_noise.index)

df_prediction_noise[["predicted_mean"]]*=7
cf.plot_combined_datasets(df_prediction_noise[["predicted_mean"]],
                        test_noise[["calls"]],
                        "2018-10-09", "2019-05-27",
                        "Q",
                        COL_DF2="0.65",
                        LINESTYLE_DF2="-",
                        LINEWIDTH_DF1=3,
                        FIGSIZE=(16, 8))

# Check model
print(df_prediction_noise)
print(df_residuals)

cf.scatter_correlation(df_prediction_noise["predicted_mean"], test_noise["calls"], X_LABEL="Prediction", Y_LABEL="Value")
cf.validate(df_prediction_noise["predicted_mean"], test_noise["calls"])"""





"""
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# 📌 Train-Test-Split (80% Training, 20% Test)
split_ratio = 0.8
split_index = int(len(df) * split_ratio)
train, test = df.iloc[:split_index], df.iloc[split_index:]

# 📌 Holt-Winters-Modell (Additiv für stabile saisonale Schwankungen)
model = ExponentialSmoothing(train['calls'], trend='add', seasonal='add', seasonal_periods=365)
model_fit = model.fit()

# 📌 Vorhersage für den Testzeitraum
forecast = model_fit.forecast(steps=len(test))

# 📌 Visualisierung der Ergebnisse
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['calls'], label='Original Data', alpha=0.6)
plt.plot(test.index, test['calls'], label='Test Data', color='green')
plt.plot(test.index, forecast, label='Holt-Winters Forecast', color='red', linestyle='dashed')
plt.legend()
plt.title('Holt-Winters Modell für tägliche Zeitreihe mit jährlicher Saisonalität')
plt.show()

df_forecast = pd.DataFrame(forecast, index=test.index)
df_forecast.rename(columns={0: "calls_predicted"}, inplace=True)

print(df_forecast)

cf.plot_combined_datasets(df_forecast[["calls_predicted"]],
                        test[["calls"]],
                        "2018-10-09", "2019-05-27",
                        "Q",
                        COL_DF2="0.65",
                        LINESTYLE_DF2="-",
                        LINEWIDTH_DF1=3,
                        FIGSIZE=(16, 8))

# Check model
cf.scatter_correlation(df_forecast["calls_predicted"], test["calls"], X_LABEL="Prediction", Y_LABEL="Value")
cf.validate(df_forecast["calls_predicted"], test["calls"])

"""


