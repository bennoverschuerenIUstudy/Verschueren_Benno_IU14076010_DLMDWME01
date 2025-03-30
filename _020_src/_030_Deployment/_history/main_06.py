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
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Beispiel: df ist bereits ein DataFrame mit der Zielspalte 'n_duty_required'

# 🔹 1️⃣ Feature Engineering

# Erstelle einen neuen DataFrame df_features für die Features
df_features = pd.DataFrame(index=df.index)

# Lag-Features (z. B. 1 Tag, 7 Tage, 30 Tage zurück)
df_features['lag_1'] = df['calls'].shift(1)  # Vorheriger Tag
df_features['lag_7'] = df['calls'].shift(7)  # 7 Tage zurück
df_features['lag_30'] = df['calls'].shift(30)  # 30 Tage zurück

# Saisonale Features (z. B. Wochentag und Monat)
df_features['day_of_week'] = df.index.dayofweek  # Wochentag (0 = Montag, 6 = Sonntag)
df_features['month'] = df.index.month  # Monat (1 = Januar, 12 = Dezember)

# Gleitende Durchschnitte (z. B. 7 Tage, 30 Tage)
df_features['rolling_mean_7'] = df['calls'].rolling(window=7).mean()  # 7-Tage-Durchschnitt
df_features['rolling_mean_30'] = df['calls'].rolling(window=30).mean()  # 30-Tage-Durchschnitt

# 7-Tage Standardabweichung
df_features['rolling_std_7'] = df['calls'].rolling(window=7).std()

# Entfernen der NaN-Werte (durch Lag-Features und gleitende Durchschnitte)
df_features.dropna(inplace=True)

# 🔹 2️⃣ Daten für Modellvorhersagen vorbereiten

# Tagesanzahl als Feature für polynomiale Regression
days = np.arange(len(df_features))  
time_series = df["calls"].loc[df_features.index].values  

# Polynomiale Features (nur linearer Trend)
degree = 1
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(days.reshape(-1, 1))

# Fourier-Features für die Saisonalität
num_harmonics = 1
X_fourier = np.column_stack([np.sin(2 * np.pi * i * days / 365) for i in range(1, num_harmonics + 1)] + [
    np.cos(2 * np.pi * i * days / 365) for i in range(1, num_harmonics + 1)
])

# Kombination der Features
X_combined = np.hstack((X_poly, X_fourier, df_features[['lag_1', 'lag_7', 'lag_30', 'day_of_week', 'month', 'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7']].values))



# 🔹 3️⃣ Trainings- und Testdatensatz aufteilen
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_combined, time_series, df_features.index, test_size=0.2, shuffle=False
)

# Speichern in DataFrames mit Datum als Index
df_train = pd.DataFrame({"date": train_indices, "calls": y_train}).set_index("date")
df_test = pd.DataFrame({"date": test_indices, "calls": y_test}).set_index("date")

# 🔹 4️⃣ Erstes XGBoost-Modell trainieren

# XGBoost 1 (Standard)
xgb_model_1 = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
xgb_model_1.fit(X_train, y_train)

# Vorhersagen des ersten XGBoost-Modells
xgb_pred_train_1 = xgb_model_1.predict(X_train)
xgb_pred_test_1 = xgb_model_1.predict(X_test)

# 🔹 5️⃣ Fehlerkorrektur mit Multiplikator

# Vorhersagen des ersten Modells anpassen, dass sie nicht kleiner als der wahre Wert sind
xgb_pred_train_1 = np.maximum(xgb_pred_train_1, y_train)
xgb_pred_test_1 = np.maximum(xgb_pred_test_1, y_test)

# Fehler aus den Vorhersagen des ersten Modells berechnen
error_train_1 = y_train - xgb_pred_train_1
error_test_1 = y_test - xgb_pred_test_1

# Berechnung des Multiplikators basierend auf dem Fehler (z.B. 1 + Fehleranteil)
multiplier_train_1 = 1 + (error_train_1 / y_train)
multiplier_test_1 = 1 + (error_test_1 / y_test)

# 🔹 6️⃣ Zweites XGBoost-Modell mit Fehlerkorrektur als Feature trainieren

# Füge die Multiplikatoren als Feature hinzu
X_train_2 = np.column_stack((X_train, multiplier_train_1))  # Füge Multiplikator als Feature hinzu
X_test_2 = np.column_stack((X_test, multiplier_test_1))  # Füge Multiplikator als Feature hinzu

# XGBoost 2 (mit Multiplikator als Feature)
xgb_model_2 = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
xgb_model_2.fit(X_train_2, y_train)

# Vorhersagen des zweiten XGBoost-Modells
xgb_pred_train_2 = xgb_model_2.predict(X_train_2)
xgb_pred_test_2 = xgb_model_2.predict(X_test_2)

# Vorhersagen des zweiten Modells anpassen, dass sie nicht kleiner als der wahre Wert sind
xgb_pred_train_2 = np.maximum(xgb_pred_train_2, y_train)
xgb_pred_test_2 = np.maximum(xgb_pred_test_2, y_test)

# 🔹 7️⃣ MSE & RMSE berechnen

# MSE & RMSE für XGBoost 1
mse_xgb_1 = mean_squared_error(y_test, xgb_pred_test_1)
rmse_xgb_1 = np.sqrt(mse_xgb_1)
rmse_percent_xgb_1 = (rmse_xgb_1 / np.mean(y_test)) * 100

# MSE & RMSE für XGBoost 2
mse_xgb_2 = mean_squared_error(y_test, xgb_pred_test_2)
rmse_xgb_2 = np.sqrt(mse_xgb_2)
rmse_percent_xgb_2 = (rmse_xgb_2 / np.mean(y_test)) * 100

# Ausgabe der Metriken
print(f"🔹 XGBoost 1 MSE: {mse_xgb_1:.4f}")
print(f"🔹 XGBoost 1 RMSE: {rmse_xgb_1:.4f}")
print(f"🔹 XGBoost 1 RMSE in Prozent: {rmse_percent_xgb_1:.2f}%")

print(f"🔹 XGBoost 2 MSE: {mse_xgb_2:.4f}")
print(f"🔹 XGBoost 2 RMSE: {rmse_xgb_2:.4f}")
print(f"🔹 XGBoost 2 RMSE in Prozent: {rmse_percent_xgb_2:.2f}%")

# 🔹 8️⃣ Plot der Ergebnisse

# Plot für Test- und Trainingsdaten (Wahre Werte vs. Vorhersagen)
fig, ax = plt.subplots(figsize=(16,4))

# Generate date range for x-axis ticks
major_locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
formatter = mdates.DateFormatter('%Y-%m-%d')

# Wahre Werte für Trainingsdaten und Testdaten
plt.plot(df_train.index, df_train['calls'], label='Wahre Werte (Training)', color='black', linewidth=1)
plt.plot(df_test.index, df_test['calls'], label='Wahre Werte (Test)', color='0.75', linewidth=1)

# Vorhersagen von XGBoost 2 für Testdaten
plt.plot(df_test.index, xgb_pred_test_2, label='XGBoost 2 Vorhersage (Test mit Fehlerkorrektur)', color='red', linestyle='-', linewidth=1)

# Titel und Labels
plt.title('Vergleich der Vorhersagen und wahren Werte mit Fehlerkorrektur (Training und Test)')
plt.xlabel('Datum')
plt.ylabel('calls')
plt.legend(loc='best')

# Set major ticks and format
ax.xaxis.set_major_locator(major_locator)
ax.xaxis.set_major_formatter(formatter)
ax.tick_params(labelsize=8)

# Add gridlines
plt.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

# Modify plot frame
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Show plot
plt.tight_layout(pad=2.0)
plt.show()





