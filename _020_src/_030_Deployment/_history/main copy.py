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
df = df.loc[df.index <= "2019-01-01"]

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
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Annahme: df ist das DataFrame mit den täglichen "calls" und dem Datumsindex

# 1️⃣ Fourier-Transformation der Zeit (saisonale Frequenzen)
def fourier_series_features(df, period, num_terms=2):
    """
    Berechnet die Fourier-Sinus- und Cosinus-Komponenten für die Saisonabhängigkeit.
    period: Periodenlänge (z.B. 365 für ein Jahr, 7 für eine Woche).
    num_terms: Anzahl der Fourier-Terme, die für die Modellierung der Saisonalität verwendet werden.
    """
    fourier_features = pd.DataFrame(index=df.index)
    for n in range(1, num_terms + 1):
        fourier_features[f'sin_{n}'] = np.sin(2 * np.pi * n * df.index.dayofyear / period)
        fourier_features[f'cos_{n}'] = np.cos(2 * np.pi * n * df.index.dayofyear / period)
    return fourier_features

# 2️⃣ Feature Engineering: Fourier-Saisonabhängigkeit (z.B. für ein Jahr)
df_features = fourier_series_features(df, period=365, num_terms=5)

# 3️⃣ Trendmodell (lineare Regression über die Zeit)
df_features['date_num'] = (df.index - df.index.min()).days
trend_model = LinearRegression()
trend_model.fit(df_features[['date_num']], df['calls'])

# Trendwerte berechnen
df_features['trend'] = trend_model.predict(df_features[['date_num']])

# Zielvariable: Anzahl der "calls"
target = df['calls'] - trend_model.predict(df_features[['date_num']])

# Feature-Matrix für die lineare Regression (Fourier-Komponenten + Trend)
X = df_features[['trend'] + [f'sin_{i}' for i in range(1, 6)] + [f'cos_{i}' for i in range(1, 6)]]

# 4️⃣ Lineare Regression: Modell erstellen und trainieren
model = LinearRegression()
model.fit(X, target)

# 5️⃣ Vorhersage auf den Trainingsdaten
y_pred_train = model.predict(X)

# Berechne den RMSE (Root Mean Squared Error) auf den Trainingsdaten
mse_train = mean_squared_error(target, y_pred_train)
rmse_train = np.sqrt(mse_train)

print(f"RMSE auf den Trainingsdaten: {rmse_train:.2f}")

# 🔹 6️⃣ Trainings- und Testdatensatz aufteilen
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X, target, df_features.index, test_size=0.2, shuffle=False
)

# Speichern in DataFrames mit Datum als Index
df_train = pd.DataFrame({"date": train_indices, "calls": y_train}).set_index("date")
df_test = pd.DataFrame({"date": test_indices, "calls": y_test}).set_index("date")

# 🔹 7️⃣ XGBoost-Modell trainieren
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# Vorhersagen auf den Testdaten
xgb_pred_test = xgb_model.predict(X_test)

# Sicherstellen, dass Vorhersagen nicht unter den tatsächlichen Zielwerten liegen
xgb_pred_test = np.maximum(xgb_pred_test, y_test)  # Vorhersagen für Testdaten auf Zielwert korrigieren

# Berechne das RMSE und das prozentuale RMSE
mse_xgb = mean_squared_error(y_test, xgb_pred_test)
rmse_xgb = np.sqrt(mse_xgb)
rmse_percent_xgb = (rmse_xgb / np.mean(y_test)) * 100

print(f"RMSE auf den Testdaten: {rmse_xgb:.2f}")
print(f"Prozentuales RMSE: {rmse_percent_xgb:.2f}%")

# 🔹 8️⃣ Visualisierung der Ergebnisse

fig, ax = plt.subplots(figsize=(16, 6))

# Wahre Werte (Training)
plt.plot(df_train.index, df_train['calls'], label='Wahre Werte (Training)', color='black', linewidth=1)

# Wahre Werte (Test)
plt.plot(df_test.index, df_test['calls'], label='Wahre Werte (Test)', color='0.75', linewidth=1)

# Vorhersagen der linearen Regression
plt.plot(df.index, y_pred_train, label='Vorhersagen (Lineare Regression)', color='red', linestyle='--', linewidth=2)

# Vorhersagen von XGBoost für Testdaten
plt.plot(df_test.index, xgb_pred_test, label='XGBoost Vorhersage', color='blue', linestyle='-', linewidth=1)

# Titel und Labels
plt.title('Lineare Regression und XGBoost Vorhersagen mit Fourier-Saisonalen Features und Trend')
plt.xlabel('Datum')
plt.ylabel('Calls')
plt.legend(loc='best')

# Formatierung der x-Achse für das Datum
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.tick_params(labelsize=8)

# Gitterlinien
plt.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

# Layout anpassen und Plot anzeigen
plt.tight_layout(pad=2.0)
plt.show()
