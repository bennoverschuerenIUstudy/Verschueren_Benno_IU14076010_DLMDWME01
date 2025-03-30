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





import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Beispiel für deine Daten (df_pred)
df_pred = df[["calls"]].copy()

# Features erstellen (Lags und saisonale Features)
df_pred['lag_1'] = df_pred['calls'].shift(1)
df_pred['lag_2'] = df_pred['calls'].shift(7)
df_pred['lag_3'] = df_pred['calls'].shift(30)
df_pred['lag_4'] = df_pred['calls'].shift(90)
df_pred['lag_5'] = df_pred['calls'].rolling(window=7).mean().shift(1)
df_pred['lag_6'] = df_pred['calls'].rolling(window=30).mean().shift(1)
df_pred['lag_7'] = df_pred['calls'].diff(1)  # Tagesdifferenz
df_pred['lag_8'] = df_pred['calls'].diff(7)  # Wochenveränderung

df_pred['lag_9'] = df_pred.index.dayofweek
df_pred['lag_10'] = (df_pred['lag_9'] >= 5).astype(int)  # Dummy für Wochenende
df_pred['lag_11'] = df_pred.index.month
df_pred['lag_12'] = df_pred.index.quarter  # Quartal

# Saisonale Features
df_pred['lag_13'] = np.sin(2 * np.pi * df_pred.index.month / 12)
df_pred['lag_14'] = np.cos(2 * np.pi * df_pred.index.month / 12)
df_pred['lag_15'] = np.sin(2 * np.pi * df_pred.index.dayofweek / 7)
df_pred['lag_16'] = np.cos(2 * np.pi * df_pred.index.dayofweek / 7)

# Gleitende Durchschnitte
df_pred['lag_17'] = df_pred['calls'].rolling(window=7).mean().shift(1)
df_pred['lag_18'] = df_pred['calls'].rolling(window=30).mean().shift(1)
df_pred['lag_19'] = df_pred['calls'].rolling(window=7).std().shift(1)

# Exponentiell gewichteter Mittelwert
df_pred['lag_20'] = df_pred['calls'].ewm(span=2).mean().shift(1)  
df_pred['lag_21'] = df_pred['calls'].ewm(span=3).mean().shift(1)  
df_pred['lag_22'] = df_pred['calls'].ewm(span=4).mean().shift(1)  

df_pred = df_pred.dropna()

# X und y definieren
X = df_pred.drop(columns=['calls']).values
y = df_pred['calls'].values

# Trainings- und Testdaten aufteilen
train_size = int(0.8 * len(df_pred))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]



"""# Define Model
model = XGBRegressor(random_state=42)

# Grid Search Parameter Grid
param_grid = {
    'n_estimators': [200, 400, 800],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [1, 2, 3],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.4, 0.6, 0.8],
    'colsample_bytree': [0.8, 1.0, 1.2]
}

# Grid Search CV
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best Parameters
best_params = grid_search.best_params_
print("Beste Parameter:", best_params)

# Train Model with Best Params
model = XGBRegressor(**best_params, random_state=42)


PRINT:
Beste Parameter: {'colsample_bytree': 1.0, 'learning_rate': 0.2, 'max_depth': 1, 'min_child_weight': 1, 'n_estimators': 800, 'subsample': 0.6}
"""




# Modell erstellen und trainieren
model = XGBRegressor(n_estimators=800, 
                     learning_rate=0.2, 
                     colsample_bytree=1.0, 
                     max_depth=1, 
                     min_child_weight=1, 
                     subsample=0.6, 
                     random_state=42)

model.fit(X_train, y_train)

# Vorhersage für Testdaten
y_pred_test = model.predict(X_test)

# Calculate the Mean Squared Error
from numpy import sqrt
from sklearn.metrics import mean_squared_error
mse = sqrt(mean_squared_error(y_test, y_pred_test)) / df["calls"].mean() * 100
print(f"Mean Squared Error: {mse:.2f}%")










# Out-of-Sample Vorhersage für zukünftige Zeitpunkte
# Hier generieren wir zukünftige Datenpunkte für die Vorhersage
out_of_sample_steps = 30  # Vorhersage für die nächsten 30 Schritte (z.B. 30 Tage)
last_known_values = df_pred.iloc[-1, :-1].values.reshape(1, -1)  # Die letzten bekannten Daten (Features)

# Liste für zukünftige Vorhersagen
future_predictions = []

for i in range(out_of_sample_steps):
    # Vorhersage für den nächsten Tag
    next_pred = model.predict(last_known_values)[0]
    future_predictions.append(next_pred)
    
    # Neue Feature-Werte für den nächsten Tag generieren
    # Wir simulieren hier, dass die Werte der Lags entsprechend verschoben werden (du kannst dies auch anpassen)
    new_features = np.roll(last_known_values, shift=-30, axis=1)
    new_features[0, -1] = next_pred  # Das Vorhergesagte wird als neues 'calls' für die nächste Iteration genommen
    last_known_values = new_features

# Plotten der Ergebnisse
plt.figure(figsize=(16, 6))

# Trainingsdaten plotten
plt.plot(df_pred.index[:train_size], y[:train_size], label="Traindata", color="black", linewidth=1)

# Testdaten (Expected) plotten
plt.plot(df_pred.index[train_size:], y_test, label="Testdata", color="0.75", linewidth=1)

# Vorhersagen auf Testdaten (rot)
plt.plot(df_pred.index[train_size:], y_pred_test, label="Predictions", color="red", linewidth=1)

# Vorhersagen (Out-of-Sample) plotten (blau)
future_dates = pd.date_range(df_pred.index[-1] + pd.Timedelta(days=1), periods=out_of_sample_steps, freq='D')
plt.plot(future_dates, future_predictions, label="Predictions out-of-sample", color="blue", linewidth=1)

# Diagramm konfigurieren
plt.legend()
plt.title("Trainingsdaten, Testdaten und Zukünftige Vorhersagen")
plt.ylabel("calls")
plt.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

# Diagramm anzeigen
plt.show()
