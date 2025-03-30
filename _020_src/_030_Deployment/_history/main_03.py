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

"""# Plot entire data
cf.plot_data(df, 
             "2016-04-01", 
             "2019-08-31", 
             "Q" )"""

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
# Offset n_duty with n_sick
df["n_duty_real"] -= df["n_sick"]

"""cf.plot_combined_datasets(df[["n_duty_real"]], 
                          df[["n_duty"]], 
                          "2016-04-01", "2019-01-01", 
                          FREQ="Q", 
                          FIGSIZE=(16,4))"""

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


# Offset the calls with the amount of drivers
df["calls_by_sby"] = df["calls"] - df["n_duty_real"] * max_corr_factor
df["calls_by_duty"] = df["calls"] - df["n_duty_real"] * max_corr_factor

# Cut the values under/over 0
# How many calls were done by standby staff 
df.loc[df["calls_by_sby"] < 0, "calls_by_sby"] = 0
# How many calls were done by duty staff
df.loc[df["calls_by_duty"] > 0, "calls_by_duty"] = 0

# Calls_by_ must also be offsetted by the max_corr_factor
df["calls_by_sby"] = (df["calls_by_sby"] / max_corr_factor).round()
df["calls_by_duty"] = (df["calls_by_duty"] / max_corr_factor).round()


# NOT IMPORTANT
# Compare sby_need with real calls_by_sby 
#cf.plot_data(df[["calls_by_sby", "dafted"]],
#            "2016-04-01", "2019-08-31", 
#             "Q",
#             (16,8))


# Check, if calls_by_sby correct
# Plot correlation of "calls" and "dafted"
"""cf.scatter_correlation(df["dafted"], 
                       df["calls_by_sby"])"""



# Plot calls_by_sby and calls_by_duty
"""cf.plot_combined_datasets(df[["calls_by_sby"]], df[["calls_by_duty"]], 
                          "2016-04-01", "2019-01-01", 
                          "Q",
                          LINESTYLE_DF2="-",
                          FIGSIZE=(16,4))"""


# FIND OFFSET FOR N_DUTY
# -----------------------

# Init values
df["n_duty_real_optimized"] = df["n_duty_real"]
df["calls_by_sby_optimized"] = df["calls_by_sby"]
n_duty_optimization = 0
pred_offset = 2200

# Iterate to find the best offset for n_duty
for i in range(1):
    # Get the difference of the medians
    n_duty_offset = (df["calls_by_sby_optimized"].abs().median() - df["calls_by_duty"].abs().median())
    # Offset n_duty
    df["n_duty_real_optimized"] += round(n_duty_offset)

    # Update calls_by_duty and calls_by_sby with the offset
    # Offset the calls with the amount of drivers
    df["calls_by_sby_optimized"] = df["calls"] - df["n_duty_real_optimized"] * max_corr_factor
    df["calls_by_duty_optimized"] = df["calls"] - df["n_duty_real_optimized"] * max_corr_factor

    # Cut the values under/over 0
    # How many calls were done by standby staff 
    df.loc[df["calls_by_sby_optimized"] < 0, "calls_by_sby_optimized"] = 0
    # How many calls were done by duty staff
    df.loc[df["calls_by_duty_optimized"] > 0, "calls_by_duty_optimized"] = 0

    # Calls_by_ must also be offsetted by the max_corr_factor
    df["calls_by_sby_optimized"] = (df["calls_by_sby_optimized"]).round() / max_corr_factor
    df["calls_by_duty_optimized"] = (df["calls_by_duty_optimized"]).round() / max_corr_factor

    # Add n_duty_offset
    n_duty_optimization += round(n_duty_offset)



print("\n\nOPTIMIZATION FOR N_DUTY\n-------------------------")
print(f"Offset:\t\t{n_duty_optimization}")
print("-------------------------")


"""cf.plot_combined_datasets(df[["n_duty_real_optimized"]], 
                          df[["n_duty_real"]], 
                          "2016-04-01", "2019-08-31", 
                          FREQ="Q",
                          LINESTYLE_DF2="-",
                          COL_DF2="0.75", 
                          FIGSIZE=(16,4))"""


"""cf.plot_combined_datasets(df[["calls_by_sby_optimized"]], df[["calls_by_duty_optimized"]], 
                          "2016-04-01", "2019-08-31", 
                          "Q",
                          LINESTYLE_DF2="-",
                          FIGSIZE=(16,4))
"""

# Check, if the calculations are correct
# Add calls to n_duty_real => how many drivers we really need (based on calls)?
#df["n_duty_required"] = df["n_duty_real"] + df["calls_by_sby_optimized"] + df["calls_by_duty_optimized"]
df["n_duty_required"] = df["calls_by_sby_optimized"] + df["calls_by_duty_optimized"]

# Get correlation of the calculated required duty and calls
"""print(cf.scatter_correlation(df["n_duty_required"], df["calls"]))
"""
# Baseline Prediction
"""cf.baseline_prediction(df, COL="calls_by_sby_optimized", FREQ="Q")
"""

# Create additive decomposition
#df_decomp = pd.DataFrame({"y": df["calls_by_sby_optimized"]})
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

# Define train/test split
split_ratio = 0.8

# Predict all decomps
df_prediction = pd.DataFrame({
     "observed": df_decomp_calls_by_sby["observed"],
     "trend": cf.XGBoost(df_decomp_calls_by_sby[["trend"]], "trend", SPLIT=split_ratio),
     "seasonal": cf.XGBoost(df_decomp_calls_by_sby[["seasonal"]], "seasonal", SPLIT=split_ratio),
     "residual": cf.XGBoost(df_decomp_calls_by_sby[["residual"]], "residual", SPLIT=split_ratio),
})

# Add columns, except "observed" and transfer NaN values
df_prediction["prediction"] = df_prediction.iloc[:, 1:].sum(axis=1, min_count=1)

train_size = int(len(df_decomp_calls_by_sby) * split_ratio)
train, test = df_decomp_calls_by_sby[:train_size].copy(), df_decomp_calls_by_sby[train_size:].copy()

# Validate
df_prediction.loc[df_prediction["prediction"] < 0, "prediction"] = 0
cf.validate(df_prediction["prediction"].dropna(), test["observed"])

#df_prediction-=pred_offset



# Cut values, below 0
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
ax[0].plot(df_prediction.index, df_prediction["trend"], label="Prediction Trend", color="red")  # Predictions
ax[0].set_ylabel("Trend")
ax[0].legend(loc="best", fontsize=8)

# Plot season
ax[1].plot(train.index, train["seasonal"], label="Trainingsdaten Seasonal", color="black")  # Trainingsdaten
ax[1].plot(test.index, test["seasonal"], label="Testdaten Seasonal", color="0.75")  # Testdaten
ax[1].plot(df_prediction.index, df_prediction["seasonal"], label="Prediction Seasonal", color="red")
ax[1].set_ylabel("Seasonal")
ax[1].legend(loc="best", fontsize=8)

# Plot residual
ax[2].plot(train.index, train["residual"], label="Trainingsdaten Residual", color="black")  # Trainingsdaten
ax[2].plot(test.index, test["residual"], label="Testdaten Residual", color="0.75")  # Testdaten
ax[2].plot(df_prediction.index, df_prediction["residual"], label="Prediction Residuals", color="red")
ax[2].set_ylabel("Residuals")
ax[2].legend(loc="best", fontsize=8)

# Plot prediction
ax[3].plot(train.index, train["observed"], label="Trainingsdaten Original", color="black")  # Trainingsdaten
ax[3].plot(test.index, test["observed"], label="Testdaten Original", color="0.75")  # Testdaten
ax[3].plot(df_prediction.index, df_prediction["prediction"], label="Prediction", color="red")
ax[3].set_ylabel("Prediction")
ax[3].legend(loc="best", fontsize=8)


# X-axis for all plots
for a in ax:
    a.xaxis.set_major_locator(major_locator)    # Ticks quarterly
    a.xaxis.set_major_formatter(formatter)      # Format
    a.tick_params(labelsize=6)                  # Fontsize
    
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
cf.scatter_correlation(df_prediction["prediction"].dropna(), test["observed"])









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