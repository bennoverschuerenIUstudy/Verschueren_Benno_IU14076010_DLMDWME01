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











from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

days = np.arange(len(df))  # Anzahl der Tage ab Startzeitpunkt
time_series = df["n_duty_required"].values  # Spalte mit Zeitreihenwerten

# Polynomiale Features für den Trend
degree = 1  # Nur linearer Trend
poly = PolynomialFeatures(degree)
X_poly = poly.fit_transform(days.reshape(-1, 1))

# Fourier-Features für die Saisonalität
num_harmonics = 100  # Anzahl der harmonischen Komponenten
X_fourier = np.column_stack([
    np.sin(2 * np.pi * i * days / 365) for i in range(1, num_harmonics + 1)
] + [
    np.cos(2 * np.pi * i * days / 365) for i in range(1, num_harmonics + 1)
])

# Kombination aus Polynom- und Fourier-Features
X_combined = np.hstack((X_poly, X_fourier))

# Trainings- und Testdatensatz aufteilen
X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
    X_combined, time_series, df.index, test_size=0.2, shuffle=False
)

# Ridge-Regression mit Regularisierung
ridge_model = Ridge(alpha=1.0).fit(X_train, y_train)

# Vorhersage mit Ridge-Regression
ridge_pred_train = ridge_model.predict(X_train)
ridge_pred_test = ridge_model.predict(X_test)

# Erstellen eines DataFrames für die Vorhersagen
ridge_predictions = np.concatenate([ridge_pred_train, ridge_pred_test])
df_pred_ridgeFourier = pd.DataFrame({"date": np.concatenate([train_indices, test_indices])})
df_pred_ridgeFourier["pred_trend_season"] = ridge_predictions

df_pred_ridgeFourier.set_index("date", inplace=True)


# Erstellen eines DataFrames für die Residuen (Differenz zwischen tatsächlichen Werten und Vorhersagen)
df_pred_ridgeFourier["residuals"] = df["n_duty_required"].loc[df_pred_ridgeFourier.index] - df_pred_ridgeFourier["pred_trend_season"]

df_residuals = df_pred_ridgeFourier[["residuals"]].loc[df_pred_ridgeFourier.index]


df_pred_XGBoost, model_residual = cf.XGBoost(df_residuals, 
                                                 target_col="residuals")

print(df_pred_XGBoost)


df_pre_pred = (df_pred_ridgeFourier["pred_trend_season"] + df_pred_XGBoost["prediction"]).to_frame("pred_n_duty_pre")

df_pre_pred["target"] = df["n_duty_required"]

df_pre_pred_XGBoost, model_residual = cf.XGBoost(df_pre_pred, 
                                                 target_col="target")
df_pre_pred_XGBoost["target"] = df["n_duty_required"]

df_pred_XGBoost, model_residual = cf.XGBoost(df_pre_pred_XGBoost, 
                                                 target_col="target")

#print(df_pre_pred_XGBoost)

print(df_pred_XGBoost)

cf.validate(df_pre_pred_XGBoost["target"], df_pre_pred_XGBoost["prediction"])
cf.validate(df_pre_pred_XGBoost["target"], df_pred_XGBoost["prediction"])

# Plotten der Ergebnisse
plt.figure(figsize=(16, 8))
plt.plot(df.index, df["n_duty_required"], label="Ridge-Testvorhersage", color="0.5", linewidth=1)
#plt.plot(train_indices, ridge_pred_train, label="Ridge-Trainingsvorhersage", color="black", linewidth=1)
#plt.plot(test_indices, ridge_pred_test, label="Ridge-Testvorhersage", color="red", linewidth=1)
#plt.plot(df_pre_pred.index, df_pre_pred["pred_n_duty_pre"], label="pre_pred", color="red", linewidth=1)
#plt.plot(df.index, df_pred_ridgeFourier["residuals"], label="Residuals", color="0.75", linewidth=1)
plt.plot(df_pred_XGBoost.index, df_pred_XGBoost["prediction"], label="Pred_Residuals", color="red", linewidth=1)

plt.xlabel("date")
plt.ylabel("Wert")
plt.title("Ridge-Regression für Zeitreihe mit linearem Trend und Saisonalität")
plt.legend()
plt.show()

