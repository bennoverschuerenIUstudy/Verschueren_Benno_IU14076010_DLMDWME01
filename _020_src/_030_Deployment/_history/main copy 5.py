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

# XGBoosting.com
# Evaluate XGBoost for Time Series Forecasting Using Walk-Forward Validation
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

# Generate a synthetic time series dataset
series = np.sin(0.1 * np.arange(200)) + np.random.randn(200) * 0.1

# Prepare data for supervised learning
#df = pd.DataFrame(series, columns=['value'])

#df = df["calls"]#.rename(columns={"calls": "value"})

df_pred = df[["calls"]].copy()
print(df_pred)


#for i in range(1, 4):
#    df_pred[f'lag_{i}'] = df_pred['calls'].shift(i)



df_pred['lag_1'] = df_pred['calls'].shift(1)
df_pred['lag_2'] = df_pred['calls'].shift(7)
df_pred['lag_3'] = df_pred['calls'].shift(30)
df_pred['lag_4'] = df_pred['calls'].shift(90)
df_pred['lag_5'] = df_pred['calls'].rolling(window=3).mean()
df_pred['lag_6'] = df_pred['calls'].rolling(window=7).mean()
df_pred['lag_7'] = df_pred['calls'].ewm(span=3).mean()  # Exponentiell gewichteter Mittelwert
df_pred['lag_8'] = df_pred['calls'].diff(1)  # Tagesdifferenz
df_pred['lag_9'] = df_pred['calls'].diff(7)  # Wochenveränderung

df_pred['lag_10'] = df_pred.index.dayofweek
df_pred['lag_11'] = (df_pred['lag_10'] >= 5).astype(int)  # Dummy für Wochenende
df_pred['lag_12'] = df_pred.index.month
df_pred['lag_13'] = df_pred.index.quarter  # Quartal
"""df_pred['lag_5'] = df_pred['calls'].shift(365)
df_pred['lag_6'] = df_pred['calls'].rolling(window=3).mean()
df_pred['lag_7'] = df_pred['calls'].rolling(window=7).mean()
df_pred['lag_8'] = df_pred['calls'].ewm(span=3).mean()  # Exponentiell gewichteter Mittelwert
df_pred['lag_9'] = df_pred['calls'].diff(1)  # Tagesdifferenz
df_pred['lag_10'] = df_pred['calls'].diff(7)  # Wochenveränderung
df_pred['lag_11'] = df_pred.index.dayofweek
df_pred['lag_12'] = (df_pred['lag_11'] >= 5).astype(int)  # Dummy für Wochenende
df_pred['lag_13'] = df_pred.index.month
df_pred['lag_14'] = df_pred.index.quarter  # Quartal
"""
print(df_pred)

df_pred = df_pred.dropna()



X = df_pred.drop(columns=['calls']).values
y = df_pred['calls'].values

# Define the number of lags and the test size for each iteration
n_lags = 800
n_test = 1

# Initialize lists to store predictions and actual values
predictions = []
actual = []

# Perform walk-forward validation
for i in range(len(X) - n_lags - n_test + 1):
    X_train, X_test = X[i:i+n_lags], X[i+n_lags:i+n_lags+n_test]
    y_train, y_test = y[i:i+n_lags], y[i+n_lags:i+n_lags+n_test]

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions.extend(y_pred)
    actual.extend(y_test)


# plot expected vs preducted
from matplotlib import pyplot
pyplot.plot(actual, label='Expected')
pyplot.plot(predictions, label='Predicted')
pyplot.legend()
pyplot.show()

# Calculate the Mean Squared Error
from numpy import mean, sqrt


mse = sqrt(mean_squared_error(actual, predictions)) / df["calls"].mean() * 100
print(f"Mean Squared Error: {mse:.4f}")