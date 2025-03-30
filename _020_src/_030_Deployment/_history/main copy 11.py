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
#df["n_duty_real"] = df["n_duty"] + df["n_sby"]

# Save the duty_real for visualisation
#df_temp = df[["n_duty"]].copy()
#df_temp["n_duty"] = df["n_duty_real"]

# Offset n_duty by n_sick
df["n_duty_real"] = df["n_duty"] - df["n_sick_modified"]

"""cf.plot_combined_datasets(df[["n_duty_real"]], 
                          df[["n_duty"]], 
                          "2016-04-01", "2019-08-31", 
                          FREQ="Q", 
                          FIGSIZE=(16,4))"""



# Initialisiere Variablen, um den besten Faktor und die höchste Korrelation zu speichern
max_corr_factor = None
highest_correlation = float('-inf')  # Kleinster möglicher Wert als Startpunkt

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
        max_corr_factor = i

# Ausgabe des besten Faktors und der höchsten Korrelation
print("\n\nGET MAX CORRELATION\n-------------------------------")
print(f"MAX_CORR_FACTOR:\t{max_corr_factor:.3f}\nCORRELATION:\t\t{highest_correlation:.3f}")
print("-------------------------------\n")

# Offset the calls with the amount of drivers
df["calls_by_sby"] = df["calls"] / max_corr_factor - df["n_duty_real"]
df["calls_by_duty"] = df["calls"] / max_corr_factor - df["n_duty_real"]
df["calls_by_dafted"] = df["calls_by_sby"] - df["n_sby"]


# Cut the values under/over 0
# How many calls were done by standby staff 
df.loc[df["calls_by_sby"] < 0, "calls_by_sby"] = 0
df.loc[df["calls_by_dafted"] < 0, "calls_by_dafted"] = 0
df.loc[df["calls_by_duty"] > 0, "calls_by_duty"] = 0


"""# Compare sby_need with real calls_by_sby 
cf.plot_data(df[["calls_by_dafted", "dafted"]],
            "2016-04-01", "2019-08-31", 
             "Q",
             (16,8))


# Check, if calls_by_sby correct
# Plot correlation of "calls" and "dafted"
cf.scatter_correlation(df["dafted"], 
                       df["calls_by_dafted"],
                       FIGSIZE=(8, 8))"""

"""# Compare sby_need with real calls_by_sby 
cf.plot_data(df[["calls_by_duty"]],
            "2016-04-01", "2019-08-31", 
             "Q",
             (16,4))"""

"""# Plot calls_by_sby and calls_by_duty
cf.plot_combined_datasets(df[["calls_by_sby"]], df[["calls_by_duty"]], 
                          "2016-04-01", "2019-08-31", 
                          "Q",
                          LINESTYLE_DF2="-",
                          FIGSIZE=(16,4))"""


df_temp = pd.DataFrame({"calls_by_sby + calls_by_duty": (df["calls_by_sby"] + df["calls_by_duty"] + df["n_duty_real"])*max_corr_factor})
"""cf.scatter_correlation(df_temp["calls_by_sby + calls_by_duty"], 
                       df["calls"])"""



# FIND OFFSET FOR EFFICIENT N_DUTY
# --------------------------------

# Define offset based on calls_by_duty and calls_by_sby
# Get the difference of the medians
n_duty_offset = (df["calls_by_sby"].abs().median() - df["calls_by_duty"].abs().median())

# Offset n_duty by the difference of the medians
df["n_duty_real_optimized"] = df["n_duty_real"] + n_duty_offset
df["n_duty_optimized"] = df["n_duty"] + n_duty_offset

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


"""cf.plot_combined_datasets(df[["n_duty_optimized"]], 
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
"""


# Check, if the calculations are correct
# Add calls to n_duty_real => how many drivers we really need (based on calls)?
df["n_duty_required"] = df["n_duty_real"] + df["calls_by_sby_optimized"] + df["calls_by_duty_optimized"]
df["n_duty_required"] = df["calls_by_sby_optimized"] + df["calls_by_duty_optimized"]

# Get correlation of the calculated required duty and calls
#print(cf.scatter_correlation(df["n_duty_required"], df["calls"]))




# Baseline Prediction
"""cf.baseline_prediction(df, COL="calls_by_sby_optimized", FREQ="Q")
"""



from prophet import Prophet
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.dates as mdates


def Fit_Predict_Prophet(df, col, CPS=0.01, SPS=10, YEARLY_SEASON=250, OFFSET=-2.5):

    # Daten für Prophet vorbereiten
    df_prophet = df[[col]].reset_index()
    df_prophet.rename(columns={'date': 'ds', col: 'y'}, inplace=True)

    # Train-Test-Split (80% Training, 20% Test)
    train_size = int(len(df_prophet) * 0.8)
    train, test = df_prophet[:train_size], df_prophet[train_size:]

    model = Prophet(changepoint_prior_scale=CPS, 
                    seasonality_prior_scale=SPS, 
                    yearly_seasonality=YEARLY_SEASON)
    
    model.add_seasonality(name='yearly', period=365, fourier_order=400)  # Jährliche Saisonalität
    
    model.fit(train)

    # Test-Vorhersagen
    test_future = test[['ds']]
    test_forecast = model.predict(test_future)

    test_forecast['yhat'] += test_forecast['yhat'].std() * OFFSET  # Verstärkt die Vorhersage

    # RMSE berechnen
    rmse = np.sqrt(mean_squared_error(test['y'], test_forecast['yhat']))
    percentage_rmse = (rmse / np.mean(test['y'])) * 100
    print(f'Prozentualer RMSE: {percentage_rmse:.2f}%')

    # MAE berechnen
    mae = mean_absolute_error(test['y'], test_forecast['yhat'])
    percentage_mae = (mae / np.mean(test['y'])) * 100
    print(f'Prozentualer MAE: {percentage_mae:.2f}%')

    # Visualisierung: Train-Test-Split
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    #ax.fill_between(test_forecast['ds'], test_forecast['yhat_lower'], test_forecast['yhat_upper'], color='red', alpha=0.2, label='Prediction Uncertainty')
    ax.plot(df_prophet['ds'], df_prophet['y'], label='Train data', color='black', linewidth=1)
    ax.plot(test['ds'], test['y'], label='Test data', color='0.75', linestyle='-', linewidth=1)
    ax.plot(test_forecast['ds'], test_forecast['yhat'], label='Prediction', color='red', linestyle='-', linewidth=1)
    ax.axvline(x=test['ds'].min(), color='black', linestyle='--', label='Train-Test-Split')

    ax.set_ylabel("Calls")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.show()

    return model


def Forecast(model, period):

    # Zukünftige Daten für die Prognose
    last_date = model.history['ds'].max()
    future_dates = pd.DataFrame({'ds': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=period, freq='D')})
    forecast = model.predict(future_dates)

    # Visualisierung des Forecasts
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.plot(model.history['ds'], model.history['y'], label='Historical Data', color='black', linewidth=1)
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='red', linestyle='-', linewidth=1)

    ax.set_ylabel("Calls")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.show()

    return forecast



model = Fit_Predict_Prophet(df, col="calls", CPS=0.01, SPS=10, YEARLY_SEASON=220, OFFSET=0.5)
#model = Fit_Predict_Prophet(df, col="n_sick_modified", CPS=0.001, SPS=10, YEARLY_SEASON=250, OFFSET=-0.25)
#forecast_df = Forecast(model, period=90)