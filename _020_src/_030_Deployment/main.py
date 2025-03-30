import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib as mpl
import numpy as np

import _020_src._global_parameters as gp
from _020_src._010_DataPrep import data_prep as dp
from _020_src._020_Modeling import modeling as m
from _020_src._030_Deployment import custom_functions as cf

df=dp.load_prepare_data(PATH="_010_data/_010_Raw/sickness_table.csv")

# Check data
dp.check_data(df)

# Plot entire data
cf.plot_data(df, 
             "2016-04-01", 
             "2019-08-31", 
             "Q" )

cf.plot_data(df[["n_sick"]], "2016-04-01", "2019-08-31", "Q", (16,4))
cf.plot_data(df[["calls"]], "2016-04-01", "2019-08-31", "Q", (16,4))
cf.plot_data(df[["n_duty"]], "2016-04-01", "2019-08-31", "Q", (16,4))
cf.plot_data(df[["n_sby"]], "2016-04-01", "2019-08-31", "Q", (16,4))
cf.plot_data(df[["sby_need"]], "2016-04-01", "2019-08-31", "Q", (16,4))
cf.plot_data(df[["dafted"]], "2016-04-01", "2019-08-31", "Q", (16,4))

# Compare sby_need with real calls_by_sby 
cf.plot_data(df[["sby_need", "dafted"]],
            "2016-04-01", "2019-08-31", 
             "Q",
             (16,8))


# Plot outliers
cf.plot_outliers(df[["n_sick"]], 
                 dp.detect_outliers_iqr, 
                 "2016-04-01", "2019-08-31", 
                 (16,4))


# Interpolate outlier
df["n_sick_modified"] = df["n_sick"]
df.at["2017-10-29", 'n_sick_modified'] = cf.linear_interpolation(df["n_sick_modified"]["2017-10-28"], df["n_sick_modified"]["2017-10-30"])

cf.plot_data(df[["n_sick_modified"]], 
             "2016-04-01", "2019-08-31", 
             "Q", 
             (16,4))

# Show reference from accidents statistics
dp.show_referenceDataAccidents()
dp.show_referencePopulation()


# Offset n_duty by n_sick
df["n_duty_real"] = df["n_duty"] - df["n_sick_modified"]

cf.plot_combined_datasets(df[["n_duty_real"]], 
                          df[["n_duty"]], 
                          "2016-04-01", "2019-08-31",
                          LINESTYLE_DF2="-", 
                          FREQ="Q", 
                          FIGSIZE=(16,4))


# --------------------------------------
# DEFINE CALLS_PER_DRIVER (cpd) 
# -> How many calls can a driver solve?
# --------------------------------------

calls_per_driver = None
highest_correlation = float('-inf')

for i in np.arange(4, 6, 0.001):
    # Define calls, which were done by standby
    df["calls_by_sby"] = df["calls"] / i - df["n_duty_real"] - df["n_sby"]
    df.loc[df["calls_by_sby"] < 0, "calls_by_sby"] = 0

    # Get correlation via custom function
    correlation = cf.get_correlation(df["calls_by_sby"], 
                                     df["dafted"])

    # Verify if current correlation is the highest
    if correlation > highest_correlation:
        highest_correlation = correlation
        calls_per_driver = i

# Print best cpd
print("\n\nGET MAX CORRELATION\n-------------------------------")
print(f"calls_per_driver:\t{calls_per_driver:.3f}\nCORRELATION:\t\t{highest_correlation:.3f}")
print("-------------------------------\n")

# Offset the calls with the amount of drivers
df["calls_by_sby"] = df["calls"] / calls_per_driver - df["n_duty_real"]
df["calls_by_duty"] = df["calls"] / calls_per_driver - df["n_duty_real"]

# Cut the values under/over 0
# How many calls were done by standby staff 
df.loc[df["calls_by_sby"] < 0, "calls_by_sby"] = 0
df.loc[df["calls_by_duty"] > 0, "calls_by_duty"] = 0

# Compare sby_need with real calls_by_sby 
cf.plot_data(df[["calls_by_sby"]],
            "2016-04-01", "2019-08-31", 
             "Q",
             (16,4))

# Compare sby_need with real calls_by_sby 
cf.plot_data(df[["calls_by_duty"]],
            "2016-04-01", "2019-08-31", 
             "Q",
             (16,4))

# Plot calls_by_sby and calls_by_duty
cf.plot_combined_datasets(df[["calls_by_sby"]], df[["calls_by_duty"]], 
                          "2016-04-01", "2019-08-31", 
                          "Q",
                          LINESTYLE_DF2="-",
                          FIGSIZE=(16,4))

# Scatter correlation
df_temp = pd.DataFrame({"calls_by_sby + calls_by_duty": (df["calls_by_sby"] + df["calls_by_duty"] + df["n_duty_real"])*calls_per_driver})
cf.scatter_correlation(df_temp["calls_by_sby + calls_by_duty"], 
                       df["calls"])


# --------------------------------
# FIND OFFSET FOR EFFICIENT N_DUTY
# --------------------------------

# Define offset based on calls_by_duty and calls_by_sby
# Get the difference of the medians
n_duty_offset = (df["calls_by_sby"].abs().median() - df["calls_by_duty"].abs().median())

# Offset n_duty by the difference of the medians
df["n_duty_real_optimized"] = df["n_duty_real"] + n_duty_offset
df["n_duty_optimized"] = df["n_duty"] + n_duty_offset

# Update calls_by_duty and calls_by_sby with the offset -> Align the middle to y=0
df["calls_by_sby_optimized"] = df["calls"] / calls_per_driver - df["n_duty_real_optimized"]
df["calls_by_duty_optimized"] = df["calls"] / calls_per_driver - df["n_duty_real_optimized"]

# How many calls were done by standby staff?
df.loc[df["calls_by_sby_optimized"] < 0, "calls_by_sby_optimized"] = 0
# How many calls were done by duty staff?
df.loc[df["calls_by_duty_optimized"] > 0, "calls_by_duty_optimized"] = 0

# Add n_duty_offset
n_duty_optimization = round(n_duty_offset)

print("\n\nOPTIMIZATION FOR N_DUTY\n-------------------------")
print(f"Offset:\t\t{n_duty_optimization}")
print("-------------------------")


cf.plot_combined_datasets(df[["n_duty_optimized"]], 
                          df[["n_duty"]], 
                          "2016-04-01", "2019-08-31", 
                          FREQ="Q",
                          LINESTYLE_DF2="-",
                          COL_DF2="black", 
                          FIGSIZE=(16,4))


cf.plot_combined_datasets(df[["calls_by_sby_optimized"]], df[["calls_by_duty_optimized"]], 
                          "2016-04-01", "2019-08-31", 
                          "Q",
                          LINESTYLE_DF2="-",
                          FIGSIZE=(16,4))


# Scatter correlation
df_temp = pd.DataFrame({"calls_by_sby_optimized + calls_by_duty_optimized": (df["calls_by_sby_optimized"] + df["calls_by_duty_optimized"] + df["n_duty_real_optimized"])*calls_per_driver})
cf.scatter_correlation(df_temp["calls_by_sby_optimized + calls_by_duty_optimized"], 
                       df["calls"])



# -----------------------------------
# FORECASTING
# -----------------------------------

# Generate of the model
# Result_model_1_Prophet (Cross-validation):    0.01 / 75.0 / 40.0 / 20
#                                               MAPE = 7.81%
# Result_model_2_Theta (Optimization):          1.0
#                                               MAPE = 12.37%

m.train_test_model(df, calls_per_driver)


# Prepare data -> cut the last 14 days
df = df.iloc[:-13]

# Forecast
df_forecast = m.forecast(DF=df, 
                         PERIOD=47, 
                         N_DUTY=1728, 
                         CPD=calls_per_driver, 
                         MAPE_1=7.81, 
                         MAPE_2=12.37, 
                         GAIN=1.96, 
                         BIAS=108.783)


fig, ax = plt.subplots(figsize=(16, 4))

ax.set_xlim(df_forecast.index.min() - pd.Timedelta(days=0.75), 
            df_forecast.index.max() + pd.Timedelta(days=0.75))

ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

ax.bar(df_forecast.index, df_forecast["forecast_sby"], color=gp._COLOR_, zorder=2)

# Label the values to its bar
for i, v in enumerate(df_forecast["forecast_sby"]):
    ax.text(df_forecast.index[i], v - 50, str(v), ha='center', fontsize=7, color="white")

ax.set_ylabel("forecast_sby")
ax.tick_params(labelsize=gp._FONTSIZE_TICK_)
plt.xticks(rotation=90)

ax.grid(False) 
ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(True)

plt.tight_layout(pad=2.0)
plt.show()
