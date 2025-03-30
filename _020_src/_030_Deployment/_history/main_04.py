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


# Create additive decomposition
df_decomp = pd.DataFrame({"y": df["n_duty_required"]})
decomp_calls_by_sby = cf.plot_seasonal_decomposition(df_decomp, COLUMN="y", PERIOD=365)

# Save decomps
df_decomp_calls_by_sby = pd.DataFrame({
    "observed": decomp_calls_by_sby.observed,
    "trend": decomp_calls_by_sby.trend,
    "seasonal": decomp_calls_by_sby.seasonal,
    "residual": decomp_calls_by_sby.resid
})


# Drop NaN values from the decomp
df_decomp_calls_by_sby.dropna(inplace=True)

# Get features
df_features = df.loc[df_decomp_calls_by_sby.index]

# Define train/test split
split_ratio = 0.8

# Predict all decomps

df_prediction=pd.DataFrame()

df_prediction["observed"] = df_decomp_calls_by_sby["observed"]

df_prediction["trend"], model_trend = cf.XGBoost(df_decomp_calls_by_sby[["trend"]], 
                                                 COL="trend", 
                                                 DF_FEATURES=df_features[["n_duty"]], 
                                                 SPLIT=split_ratio)

df_prediction["seasonal"], model_seasonal = cf.XGBoost(df_decomp_calls_by_sby[["seasonal"]], 
                                                 COL="seasonal", 
                                                 DF_FEATURES=df_features[["n_duty"]], 
                                                 SPLIT=split_ratio)

df_prediction["residual"], model_residual = cf.XGBoost(df_decomp_calls_by_sby[["residual"]], 
                                                 COL="residual", 
                                                 DF_FEATURES=df_features[["n_duty"]], 
                                                 SPLIT=split_ratio)



# Add columns, except "observed" and transfer NaN values
df_prediction["prediction"] = df_prediction.iloc[:, 1:].sum(axis=1, min_count=1)

train_size = int(len(df_decomp_calls_by_sby) * split_ratio)
train, test = df_decomp_calls_by_sby[:train_size].copy(), df_decomp_calls_by_sby[train_size:].copy()



# Validate
cf.validate(df_prediction["prediction"].dropna(), test["observed"])


# Create Prediction_Output
df_prediction["prediction"] -= df["n_duty_real"]
train["observed"] -= df["n_duty_real"]
test["observed"] -= df["n_duty_real"]

# Cut values, below 0
df_prediction.loc[df_prediction["prediction"] < 0, "prediction"] = 0
train.loc[train["observed"] < 0, "observed"] = 0
test.loc[test["observed"] < 0, "observed"] = 0


# Set timeline (quarterly)
major_locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
formatter = mdates.DateFormatter('%Y-%m-%d')

#ax.plot(result.index, train["y"], label="Trainingdata", color="black")
fig, ax = plt.subplots(nrows=4, sharex=True, figsize=(16, 12))  # 3 Zeilen, gemeinsame x-Achse

# Plot trend
ax[0].plot(train.index, train["trend"], label="Trainingsdaten Trend", color="black")  # Trainingsdaten
ax[0].plot(test.index, test["trend"], label="Testdaten Trend", color="0.75")  # Testdaten
ax[0].plot(df_prediction.index, df_prediction["trend"], label="Prediction Trend", color="#D85555")  # Predictions
ax[0].set_ylabel("Trend")
ax[0].legend(loc="upper left", fontsize=8)

# Plot season
ax[1].plot(train.index, train["seasonal"], label="Trainingsdaten Seasonal", color="black")  # Trainingsdaten
ax[1].plot(test.index, test["seasonal"], label="Testdaten Seasonal", color="0.75")  # Testdaten
ax[1].plot(df_prediction.index, df_prediction["seasonal"], label="Prediction Seasonal", color="#D85555")
ax[1].set_ylabel("Seasonal")
ax[1].legend(loc="upper left", fontsize=8)

# Plot residual
ax[2].plot(train.index, train["residual"], label="Trainingsdaten Residual", color="black")  # Trainingsdaten
ax[2].plot(test.index, test["residual"], label="Testdaten Residual", color="0.75")  # Testdaten
ax[2].plot(df_prediction.index, df_prediction["residual"], label="Prediction Residuals", color="#D85555")
ax[2].set_ylabel("Residuals")
ax[2].legend(loc="upper left", fontsize=8)

# Plot prediction
ax[3].plot(train.index, train["observed"], label="Trainingsdaten Original", color="black")  # Trainingsdaten
ax[3].plot(test.index, test["observed"], label="Testdaten Original", color="0.75")  # Testdaten
ax[3].plot(df_prediction.index, df_prediction["prediction"], label="Prediction", color="#D85555")
ax[3].set_ylabel("Prediction")
ax[3].legend(loc="upper left", fontsize=8)


# X-axis for all plots
for a in ax:
    a.xaxis.set_major_locator(major_locator)    # Ticks quarterly
    a.xaxis.set_major_formatter(formatter)      # Format
    a.tick_params(labelsize=8)                  # Fontsize
    
    # Add grid
    a.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)
    
    # Define frame
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)

# X-axis label
plt.xlabel("date")

# Show
plt.tight_layout()
plt.show()


# Scatter correlations
"""cf.scatter_correlation(df_prediction["seasonal"].dropna(), test["seasonal"], X_LABEL="prediction_seasonal")
cf.scatter_correlation(df_prediction["residual"].dropna(), test["residual"], X_LABEL="prediction_residual")"""
#cf.scatter_correlation(df_prediction["prediction"].dropna(), test["observed"])


start_date = '2015-05-01'
end_date = '2018-06-01'
n_duty = 1900  # Beispielwert für n_duty


"""print(df_prediction[["trend"]])
print(df_features[["n_duty"]])
"""


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

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 🔹 1️⃣ Feature Engineering (wie vorher, aber ohne Fehlerkorrektur)
df_features = pd.DataFrame(index=df.index)

# Lag-Features
df_features['lag_1'] = df['calls'].shift(1)
df_features['lag_7'] = df['calls'].shift(7)
df_features['lag_30'] = df['calls'].shift(30)

# Saisonale Features
df_features['day_of_week'] = df.index.dayofweek
df_features['month'] = df.index.month

# Gleitende Durchschnitte
df_features['rolling_mean_7'] = df['calls'].rolling(window=7).mean()
df_features['rolling_mean_30'] = df['calls'].rolling(window=30).mean()
df_features['rolling_std_7'] = df['calls'].rolling(window=7).std()

# Entfernen von NaN-Werten
df_features.dropna(inplace=True)

# 🔹 2️⃣ Daten für Modellvorhersagen vorbereiten
target = df["calls"].loc[df_features.index].values  
features = df_features[['lag_1', 'lag_7', 'lag_30', 'day_of_week', 'month', 'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7']].values

# 🔹 3️⃣ Trainings- und Testdatensatz aufteilen
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, shuffle=False)

# 🔹 4️⃣ XGBoost-Modell trainieren
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

# 🔹 5️⃣ Prognose für das nächste Jahr (365 Tage)
last_date = df_features.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=365, freq='D')

# Erstelle Features für die Zukunft
future_features = pd.DataFrame(index=future_dates)

# Lag-Features für zukünftige Daten (setzen auf letzte bekannten Werte)
future_features['lag_1'] = df['calls'].iloc[-1]  # Letzter bekannter Wert
future_features['lag_7'] = df['calls'].iloc[-7]  # Wert von vor 7 Tagen
future_features['lag_30'] = df['calls'].iloc[-30]  # Wert von vor 30 Tagen

# Saisonale Features
future_features['day_of_week'] = future_features.index.dayofweek
future_features['month'] = future_features.index.month

# Gleitende Durchschnitte (mit letzten bekannten Werten)
future_features['rolling_mean_7'] = df['calls'].rolling(window=7).mean().iloc[-1]
future_features['rolling_mean_30'] = df['calls'].rolling(window=30).mean().iloc[-1]
future_features['rolling_std_7'] = df['calls'].rolling(window=7).std().iloc[-1]

# Vorhersagen für die Zukunft
future_features_values = future_features[['lag_1', 'lag_7', 'lag_30', 'day_of_week', 'month', 'rolling_mean_7', 'rolling_mean_30', 'rolling_std_7']].values
future_predictions = xgb_model.predict(future_features_values)

# 🔹 6️⃣ Visualisierung der Prognose für das nächste Jahr
fig, ax = plt.subplots(figsize=(16,4))

# Vorhersagen für das nächste Jahr (365 Tage)
plt.plot(future_dates, future_predictions, label='Vorhersage für das nächste Jahr', color='blue', linestyle='--', linewidth=2)

# Wahre Werte für Trainingsdaten
plt.plot(df.index, df['calls'], label='Wahre Werte (Historische Daten)', color='black', linewidth=1)

# Titel und Labels
plt.title('XGBoost Vorhersage für das nächste Jahr')
plt.xlabel('Datum')
plt.ylabel('calls')
plt.legend(loc='best')

# Setze Haupt-Ticks und Formatierung
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.tick_params(labelsize=8)

# Füge Gitterlinien hinzu
plt.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

# Zeige den Plot
plt.tight_layout(pad=2.0)
plt.show()
