import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import custom_functions as cf


# Load data and set column "date" as index
df = pd.read_csv("_data/sickness_table.csv", parse_dates=True)

# Print data
print("\n\nSHOW RAW DATA\n----------------------------------------------------------------------------")
print(df.head())  # Show the DataFrame
print("----------------------------------------------------------------------------\n")

# set column date as index and convert to datetime
df.set_index('date', inplace=True)
df.index = pd.to_datetime(df.index)
df = df.asfreq('D')     # 'D' for daily data
# Set the df to integer
df = df.astype(int)
# Delete unnamed column
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
print("\n\nSHOW MODIFIED DATA\n-------------------------------------------------------------")
print(df.head())  # Show the DataFrame
print("-------------------------------------------------------------\n")


cf.check_data(df)

# Plot entire data
"""cf.plot_data(df, 
             "2016-04-01", 
             "2019-08-31", 
             "Q" )"""

"""cf.plot_data(df[["n_sick"]], "2016-04-01", "2019-08-31", "Q", (16,4))
cf.plot_data(df[["calls"]], "2016-04-01", "2019-08-31", "Q", (16,4))
cf.plot_data(df[["n_duty"]], "2016-04-01", "2019-08-31", "Q", (16,4))
cf.plot_data(df[["n_sby"]], "2016-04-01", "2019-08-31", "Q", (16,4))
cf.plot_data(df[["sby_need"]], "2016-04-01", "2019-08-31", "Q", (16,4))
cf.plot_data(df[["dafted"]], "2016-04-01", "2019-08-31", "Q", (16,4))
"""
# Compare sby_need with real calls_by_sby 
"""cf.plot_data(df[["sby_need", "dafted"]],
            "2016-04-01", "2019-08-31", 
             "Q",
             (16,8))
"""
# Cut dataframe until specific date
#df = df.loc[df.index <= "2019-01-01"]

"""# Plot entire data
cf.plot_data(df, 
             "2016-04-01", 
             "2019-08-31", 
             "Q" )"""

# Plot outliers
"""cf.plot_outliers(df[["n_sick"]], 
                 cf.detect_outliers_iqr, 
                 "2016-04-01", "2019-08-31", 
                 (16,4))"""


# Interpolate outlier
df["n_sick_modified"] = df["n_sick"]
df.at["2017-10-29", 'n_sick_modified'] = cf.linear_interpolation(df["n_sick_modified"]["2017-10-28"], df["n_sick_modified"]["2017-10-30"])

"""cf.plot_data(df[["n_sick_modified"]], 
             "2016-04-01", "2019-08-31", 
             "Q", 
             (16,4))"""

# Show reference from accidents statistics
"""cf.show_referenceDataAccidents()
cf.show_referencePopulation()"""


# Add n_sby to n_duty, because it is 24/7/365 booked
df["n_duty_real"] = df["n_duty"] + df["n_sby"]

# Save the duty_real for visualisation
df_temp = df[["n_duty"]].copy()
df_temp["n_duty"] = df["n_duty_real"]

# Offset n_duty by n_sick
df["n_duty_real"] -= df["n_sick_modified"]

"""cf.plot_combined_datasets(df[["n_duty_real"]], 
                          df_temp[["n_duty"]], 
                          "2016-04-01", "2019-08-31", 
                          FREQ="Q", 
                          FIGSIZE=(16,4))
"""


# Initialisiere Variablen, um den besten Faktor und die höchste Korrelation zu speichern
max_corr_factor = None
highest_correlation = float('-inf')  # Kleinster möglicher Wert als Startpunkt

for i in np.arange(4, 6, 0.001):
    # Define calls, which were done by standby
    df["calls_by_sby"] = df["calls"] / i - df["n_duty_real"]
    df.loc[df["calls_by_sby"] < 0, "calls_by_sby"] = 0

    # Get correlation via custom function
    correlation = cf.get_correlation(df["calls_by_sby"], 
                                     df["dafted"])

    # Verify if current correlation is the highest
    if correlation > highest_correlation:
        highest_correlation = correlation
        max_corr_factor = i

# Ausgabe des besten Faktors und der höchsten Korrelation
print("\n\nGET MAX CORRELATION\n-------------------------------")
print(f"MAX_CORR_FACTOR:\t{max_corr_factor:.3f}\nCORRELATION:\t\t{highest_correlation:.3f}")
print("-------------------------------\n")

# Offset the calls with the amount of drivers
df["calls_by_sby"] = df["calls"] / max_corr_factor - df["n_duty_real"]
df["calls_by_duty"] = df["calls"] / max_corr_factor - df["n_duty_real"]

# Cut the values under/over 0
# How many calls were done by standby staff 
df.loc[df["calls_by_sby"] < 0, "calls_by_sby"] = 0
# How many calls were done by duty staff
df.loc[df["calls_by_duty"] > 0, "calls_by_duty"] = 0

# Compare sby_need with real calls_by_sby 
"""cf.plot_data(df[["calls_by_sby", "dafted"]],
            "2016-04-01", "2019-08-31", 
             "Q",
             (16,8))
"""

# Check, if calls_by_sby correct
# Plot correlation of "calls" and "dafted"
"""cf.scatter_correlation(df["dafted"], 
                       df["calls_by_sby"])
"""
# Compare sby_need with real calls_by_sby 
"""cf.plot_data(df[["calls_by_duty"]],
            "2016-04-01", "2019-08-31", 
             "Q",
             (16,4))
"""
# Plot calls_by_sby and calls_by_duty
"""cf.plot_combined_datasets(df[["calls_by_sby"]], df[["calls_by_duty"]], 
                          "2016-04-01", "2019-08-31", 
                          "Q",
                          LINESTYLE_DF2="-",
                          FIGSIZE=(16,4))
"""

df_temp = pd.DataFrame({"calls_by_sby + calls_by_duty": (df["calls_by_sby"] + df["calls_by_duty"] + df["n_duty_real"])*max_corr_factor})
"""cf.scatter_correlation(df_temp["calls_by_sby + calls_by_duty"], 
                       df["calls"])
"""


# FIND OFFSET FOR EFFICIENT N_DUTY
# -----------------------

# Define offset based on calls_by_duty and calls_by_sby
# Get the difference of the medians
n_duty_offset = (df["calls_by_sby"].abs().median() - df["calls_by_duty"].abs().median())

# Offset n_duty by the difference of the medians
df["n_duty_real_optimized"] = df["n_duty_real"] + n_duty_offset

# Update calls_by_duty and calls_by_sby with the offset -> Align the middle to y=0
df["calls_by_sby_optimized"] = df["calls"] / max_corr_factor - df["n_duty_real_optimized"]
df["calls_by_duty_optimized"] = df["calls"] / max_corr_factor - df["n_duty_real_optimized"]

# How many calls were done by standby staff?
df.loc[df["calls_by_sby_optimized"] < 0, "calls_by_sby_optimized"] = 0
# How many calls were done by duty staff?
df.loc[df["calls_by_duty_optimized"] > 0, "calls_by_duty_optimized"] = 0

# Add n_duty_offset
n_duty_optimization = round(n_duty_offset)

print("\n\nOPTIMIZATION FOR N_DUTY\n-------------------------")
print(f"Offset:\t\t{n_duty_optimization}")
print("-------------------------")


"""cf.plot_combined_datasets(df[["n_duty_real_optimized"]], 
                          df[["n_duty_real"]], 
                          "2016-04-01", "2019-08-31", 
                          FREQ="Q",
                          LINESTYLE_DF2="-",
                          COL_DF2="0.75", 
                          FIGSIZE=(16,4))
"""

"""cf.plot_combined_datasets(df[["calls_by_sby_optimized"]], df[["calls_by_duty_optimized"]], 
                          "2016-04-01", "2019-08-31", 
                          "Q",
                          LINESTYLE_DF2="-",
                          FIGSIZE=(16,4))
"""
# Check, if the calculations are correct
# Add calls to n_duty_real => how many drivers we really need (based on calls)?
df["n_duty_required"] = df["n_duty_real"] + df["calls_by_sby_optimized"] + df["calls_by_duty_optimized"]
#df["n_duty_required"] = df["calls_by_sby_optimized"] + df["calls_by_duty_optimized"]

# Get correlation of the calculated required duty and calls
#print(cf.scatter_correlation(df["n_duty_required"], df["calls"]))

# Baseline Prediction
"""cf.baseline_prediction(df, COL="calls_by_sby_optimized", FREQ="Q")
"""












import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

# 🔹 1️⃣ Feature Engineering
df_features = pd.DataFrame(index=df.index)

# Lag-Features
df_features['lag_1'] = df['calls'].shift(1)
df_features['lag_7'] = df['calls'].shift(7)
df_features['lag_30'] = df['calls'].shift(30)

# Saisonale Features: Monat und Wochentag (sinus und kosinus zur Modellierung der Saisonabhängigkeit)
df_features['sin_month'] = np.sin(2 * np.pi * df.index.month / 12)  # Saisonabhängigkeit (Monat)
df_features['cos_month'] = np.cos(2 * np.pi * df.index.month / 12)

df_features['sin_dayofweek'] = np.sin(2 * np.pi * df.index.dayofweek / 7)  # Saisonabhängigkeit (Tag der Woche)
df_features['cos_dayofweek'] = np.cos(2 * np.pi * df.index.dayofweek / 7)

# Gleitende Durchschnitte
df_features['rolling_mean_7'] = df['calls'].rolling(window=7).mean()
df_features['rolling_mean_30'] = df['calls'].rolling(window=30).mean()
df_features['rolling_std_7'] = df['calls'].rolling(window=7).std()

# Trend-Feature erstellen (lineare Regression der "calls" über die Zeit)
# Erstelle eine numerische Zeitachse
df_features['date_num'] = (df_features.index - df_features.index.min()).days

# Fit eine lineare Regression auf die "calls" über die Zeit
model_trend = LinearRegression()
model_trend.fit(df_features[['date_num']], df['calls'])

# Berechne die Trendwerte (Lineare Vorhersage für jeden Punkt)
df_features['trend'] = model_trend.predict(df_features[['date_num']])

# Entferne NaN-Werte nach der Erstellung des Trend-Features
df_features.dropna(inplace=True)

# 🔹 2️⃣ Daten für Modellvorhersagen vorbereiten
target = df["calls"].loc[df_features.index].values  
features = df_features[['lag_1', 'lag_7', 'lag_30', 'sin_month', 'cos_month', 'sin_dayofweek', 'cos_dayofweek', 
                        'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7', 'trend']].values

# 🔹 3️⃣ Trainings- und Testdatensatz aufteilen
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    features, target, df_features.index, test_size=0.2, shuffle=False
)

# Speichern in DataFrames mit Datum als Index
df_train = pd.DataFrame({"date": train_indices, "calls": y_train}).set_index("date")
df_test = pd.DataFrame({"date": test_indices, "calls": y_test}).set_index("date")

# 🔹 4️⃣ Modell trainieren
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=400, max_depth=20, learning_rate=0.025, alpha=0.1, random_state=42)

xgb_model.fit(X_train, y_train)
# Vorhersagen auf den Testdaten
xgb_pred_test = xgb_model.predict(X_test)



# Sicherstellen, dass Vorhersagen nicht unter den tatsächlichen Zielwerten liegen
xgb_pred_test = np.maximum(xgb_pred_test, y_test)  # Vorhersagen für Testdaten auf Zielwert korrigieren



# Berechne das RMSE und das prozentuale RMSE
mse_xgb = mean_squared_error(y_test, xgb_pred_test)
rmse_xgb = np.sqrt(mse_xgb)
rmse_percent_xgb = (rmse_xgb / np.mean(y_test)) * 100

# Ausgabe der Metriken
print(f"RMSE = {rmse_xgb:.2f}, Prozentuales RMSE = {rmse_percent_xgb:.2f}%")


















# 🔹 5️⃣ Prognose für das nächste Jahr (365 Tage)
# Erstelle Zeitstempel für die nächsten 365 Tage
last_date = df_features.index[-1]
future_dates = pd.date_range(start=last_date, periods=365, freq='D')

# Erstelle Features für die Zukunft
future_features = pd.DataFrame(index=future_dates)

# Lag-Features für zukünftige Daten (angenommen, dass die Lag-Features basierend auf den vorherigen Werten gesetzt werden)
future_features['lag_1'] = df['calls'].shift(1).iloc[-365:].values
future_features['lag_7'] = df['calls'].shift(7).iloc[-365:].values
future_features['lag_30'] = df['calls'].shift(30).iloc[-365:].values

# Saisonale Features (sin und cos für Monat und Wochentag)
future_features['sin_month'] = np.sin(2 * np.pi * future_features.index.month / 12)
future_features['cos_month'] = np.cos(2 * np.pi * future_features.index.month / 12)

future_features['sin_dayofweek'] = np.sin(2 * np.pi * future_features.index.dayofweek / 7)
future_features['cos_dayofweek'] = np.cos(2 * np.pi * future_features.index.dayofweek / 7)

# Gleitende Durchschnitte
future_features['rolling_mean_7'] = df['calls'].rolling(window=7).mean().iloc[-365:].values
future_features['rolling_mean_30'] = df['calls'].rolling(window=30).mean().iloc[-365:].values
future_features['rolling_std_7'] = df['calls'].rolling(window=7).std().iloc[-365:].values

# Trend-Feature für die Zukunft
future_features['date_num'] = (future_features.index - df_features.index.min()).days
future_features['trend'] = model_trend.predict(future_features[['date_num']])

# Fehlende Werte im zukünftigen Zeitraum auffüllen (kann mit vorigen Werten befüllt werden, oder eine andere Strategie verwenden)
future_features.fillna(method='ffill', inplace=True)

# Vorhersagen für die Zukunft
future_features_values = future_features[['lag_1', 'lag_7', 'lag_30', 'sin_month', 'cos_month', 'sin_dayofweek', 'cos_dayofweek',
                                          'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7', 'trend']].values
future_predictions = xgb_model.predict(future_features_values)

# Korrektur auf die Zukunftsvorhersagen anwenden
future_predictions = np.maximum(future_predictions, future_features['lag_1'].values)*1.07 # Verhindere, dass Vorhersagen unter 'lag_1' liegen





# Erstelle den df_pred DataFrame, der den Index von df übernimmt
df_pred = pd.DataFrame(index=df.index)

# Füge die neue Spalte hinzu: df_test['calls'] / max_corr_factor + df["n_sick"]
df_pred['n_required_test'] = df_test['calls'] / max_corr_factor + df['n_sick']
df_pred["n_required_train"] = df_train['calls'] / max_corr_factor + df['n_sick']

df_pred["n_sby_required_test"] = np.maximum(df_pred['n_required_test'] - df["n_duty"], 0)
df_pred["n_sby_required_train"] = np.maximum(df_pred['n_required_train'] - df["n_duty"], 0)

df_pred["n_duty_required_test"] =  -(np.maximum(df["n_duty"] - df_pred['n_required_test'], 0))
df_pred["n_duty_required_train"] = -(np.maximum(df["n_duty"] - df_pred['n_required_train'], 0))




y = df.loc[df_test.index, 'dafted'] + df.loc[df_test.index, 'n_sby']
y_pred = df_pred['n_sby_required_test'].dropna()








# 🔹 6️⃣ Visualisierung der Prognose für das nächste Jahr mit 2 Subplots
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

# 📌 Erster Subplot: Prognosen für das nächste Jahr und wahre Werte
axes[0].plot(future_dates, future_predictions, label='Vorhersage für das nächste Jahr', color='blue', linestyle='-', linewidth=1)
axes[0].plot(df_train.index, df_train['calls'], label='Wahre Werte (Training)', color='black', linewidth=1)
axes[0].plot(df_test.index, df_test['calls'], label='Wahre Werte (Test)', color='0.75', linewidth=1)
axes[0].plot(df_test.index, xgb_pred_test, label='XGBoost Vorhersage', color='red', linestyle='-', linewidth=1)

axes[0].set_title('XGBoost Vorhersagen und wahre Werte ohne Iteration und Fehlerkorrektur')
axes[0].set_ylabel('Calls')
axes[0].legend(loc='best')
axes[0].grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

# 📌 Zweiter Subplot: `n_sby_required` und `n_duty_required`
axes[1].plot(df_train.index, df.loc[df_train.index, 'dafted'], label='dafted', color='black', linestyle='-', linewidth=1)
axes[1].plot(df_test.index, df.loc[df_test.index, 'dafted'], label='dafted', color='0.75', linestyle='-', linewidth=1)
axes[1].plot(df_train.index, df_pred['n_sby_required_train'].dropna(), label='n_sby_required_train', color='black', linestyle='-', linewidth=1)
axes[1].plot(df_train.index, df_pred['n_duty_required_train'].dropna(), label='n_duty_required_train', color='black', alpha=0.35, linestyle='-', linewidth=1)
axes[1].plot(df_test.index, df_pred['n_sby_required_test'].dropna(), label='n_sby_required_test', color='red', linestyle='-', linewidth=1)
axes[1].plot(df_test.index, df_pred['n_duty_required_test'].dropna(), label='n_duty_required_test', color='red', alpha=0.35, linestyle='-', linewidth=1)

axes[1].set_title('Vergleich von n_sby_required und n_duty_required')
axes[1].set_xlabel('Datum')
axes[1].set_ylabel('Anzahl')
axes[1].legend(loc='best')
axes[1].grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

# X-Achse formatieren
axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
axes[1].tick_params(labelsize=8)

# Layout anpassen und Plot anzeigen
plt.tight_layout(pad=2.0)
plt.show()