import itertools
import numpy as np
from numpy import mean, sqrt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error

import time
from darts import TimeSeries
from darts.models import Theta
from darts.metrics import mape

import _020_src._global_parameters as gp
from _020_src._010_DataPrep import data_prep as dp
from _020_src._020_Modeling import modeling as m
from _020_src._030_Deployment import custom_functions as cf


def optimize_theta(DF, COL, SPLIT=0.8):
    """
    Find the best parameter theta based on lowes MAPE

    Param:
    - DF:       DataFrame
    - COL:      column
    - SPLIT:    Splitratio (train/test)

    Return:
    - best_model:   Best model
    - df_forecast:  Dataframe with predicted values 
    - best_mape:    MAPE   
    """

    # Prepare data for darts
    series = TimeSeries.from_dataframe(DF, value_cols=COL)

    # Train/Test Split
    series_train, series_test = series.split_after(SPLIT)

    # Parameter
    results = []

    # Get best parameter
    for p in np.arange(0.005, 1.005, 0.005):
        model = Theta(seasonality_period=365, theta=p)
        
        model.fit(series_train)
        
        forecast_test = model.predict(len(series_test))
        
        # Get MAPE for param
        error = mape(series_test, forecast_test)

        # Append to the list
        results.append((p, error))

    # Get param with less MAPE
    sorted_results = sorted(results, key=lambda x: x[1])
    best_param = sorted_results[0]
    best_mape=best_param[1]
    print("\nBest Param for model_2 (Theta):\n-------------------------------")
    print(f"theta:\t{best_param[0]}\nMAPE:\t{best_mape:.2f}%")
    

    # ---------------
    # Show best model
    # ---------------

    best_model = Theta(seasonality_period=365, theta=best_param[0])
    best_model.fit(series_train)
    best_forecast = best_model.predict(len(series_test))


    # Create df for visualization
    df_train, df_test = dp.train_test_split(DF, SPLIT)
    df_forecast = best_forecast.to_dataframe()

    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=(16, 4))

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.plot(df_train.index, df_train[COL], label='Train data', color='black', linewidth=1)
    ax.plot(df_test.index, df_test[COL], label='Test data', color='0.75', linestyle='-', linewidth=1)
    ax.plot(df_forecast.index, df_forecast[COL], label='Prediction', color=gp._COLOR_, linestyle='-', linewidth=1)
    ax.axvline(x=df_test.index.min(), color='black', linestyle='--', label='Train-Test-Split')

    ax.set_ylabel("n_sick_modified")
    ax.legend(loc="upper left", fontsize=gp._FONTSIZE_LEGEND_, frameon=True)
    
    ax.tick_params(labelsize=gp._FONTSIZE_TICK_)
    
    ax.grid(False)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)
    #ax.grid(True, axis='y', linestyle='--', color='gray', linewidth=0.6)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    plt.tight_layout(pad=2.0)
    plt.show()


    # Convert to dataframe
    df_results = pd.DataFrame(results, columns=["Theta", "MAPE"])


    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df_results["Theta"], df_results["MAPE"], label="MAPE", color=gp._COLOR_, linewidth=1.5)

    ax.tick_params(labelsize=gp._FONTSIZE_TICK_)

    ax.grid(False)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)
    #ax.grid(True, axis='y', linestyle='--', color='gray', linewidth=0.6)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.set_title("MAPE vs Theta", fontsize=10)

    ax.set_xlabel("Theta", fontsize=10)
    ax.set_ylabel("MAPE", fontsize=10)

    ax.set_ylim(0, 100)

    plt.tight_layout(pad=2.0)
    plt.show()

    return best_model, df_forecast[[COL]], best_mape


def fit_theta(DF, COL, THETA):
    """
    Fit the Theta model with right parameter

    Param:
    - DF:       Dataframe
    - COL:      Column
    - THETA:    Parameter for model

    Return:
    - model
    """

    series = TimeSeries.from_dataframe(DF, value_cols=COL)

    model = Theta(seasonality_period=365, theta=THETA)
    model.fit(series)

    return model


def crossvalidate_prophet(DF, COL, PARAM_GRID, SPLIT=0.8):
    """
    Crossvalidation for Prophet model and visualize the process

    Params:
    - DF:           Dataframe
    - COL:          Column
    - PARAM_GRID:   Params to crossvalidate
    - SPLIT:        Splitratio (Train/Test)
    
    Return:
    - best_model
    - test_forecast[['yhat']]
    - mape
    """

    # Prepare data for Prophet
    df_prophet = DF[[COL]].reset_index()
    df_prophet.rename(columns={'date': 'ds', COL: 'y'}, inplace=True)
    
    # Train/Test Split
    df_train, df_test = dp.train_test_split(df_prophet, SPLIT)

    # Create hyperparameter varations
    all_params = [dict(zip(PARAM_GRID.keys(), v)) for v in itertools.product(*PARAM_GRID.values())]
    results = []

    # Init params for Prophet
    for params in all_params:
        model = Prophet(
            changepoint_prior_scale=params['changepoint_prior_scale'],
            seasonality_prior_scale=params['seasonality_prior_scale'],
            yearly_seasonality=params['yearly_seasonality'],
        )
        model.add_seasonality(name='yearly', period=365, fourier_order=params['fourier_order'])

        model.fit(df_train)

        # Cross-Validation
        df_cv = cross_validation(model, 
                                 initial='800 days', 
                                 period='2 days', 
                                 horizon='90 days')

        df_p = performance_metrics(df_cv)

        # Get Mean Average Percentage Error
        mape = mean_absolute_percentage_error(df_cv['y'], df_cv['yhat'])*100
        best_mape=mape

        # Save results
        results.append({
            'changepoint_prior_scale': params['changepoint_prior_scale'],
            'seasonality_prior_scale': params['seasonality_prior_scale'],
            'yearly_seasonality': params['yearly_seasonality'],
            'fourier_order': params['fourier_order'],
            'RMSE': df_p['rmse'].mean(),
            'MAPE': best_mape
        })

    # Save to dataframe
    results_df = pd.DataFrame(results)
    
    # Get best parameter by MAPE
    best_params = results_df.sort_values(by='MAPE').iloc[0]
    print(f'\nBest Param for model_1 (Prophet):\n---------------------------------\n{best_params}')

    # Train with best param
    best_model = Prophet(
        changepoint_prior_scale=best_params['changepoint_prior_scale'],
        seasonality_prior_scale=best_params['seasonality_prior_scale'],
        yearly_seasonality=best_params['yearly_seasonality'],
    )
    best_model.add_seasonality(name='yearly', period=365, fourier_order=best_params['fourier_order'])

    best_model.fit(df_train)

    # Predict Testdata
    test_future = df_test[['ds']]
    test_forecast = best_model.predict(test_future)

    # Plot
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.plot(df_prophet['ds'], df_prophet['y'], label='Train data', color='black', linewidth=1)
    ax.plot(df_test['ds'], df_test['y'], label='Test data', color='0.75', linestyle='-', linewidth=1)
    ax.plot(test_forecast['ds'], test_forecast['yhat'], label='Prediction', color='red', linestyle='-', linewidth=1)
    ax.axvline(x=df_test['ds'].min(), color='black', linestyle='--', label='Train-Test-Split')

    ax.set_ylabel("Calls")
    ax.legend(loc="best", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)
    #ax.grid(True, axis='y', linestyle='--', color='gray', linewidth=0.6)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    plt.tight_layout(pad=2.0)
    plt.show()

    # Heatmap
    plt.figure(figsize=(10, 4))
    sns.heatmap(results_df.pivot_table(index="changepoint_prior_scale", 
                                        columns="seasonality_prior_scale", 
                                        values="MAPE", 
                                        aggfunc='mean',
                                        ),
                cmap="Reds_r", annot=True, fmt=".4f")
    plt.tick_params(labelsize=8)
    plt.title("MAPE für verschiedene Changepoint- & Seasonality-Werte")
    plt.show()


    # Yearly Seasonality
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="yearly_seasonality", y="MAPE", data=results_df)
    plt.tick_params(labelsize=8)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    plt.show()

    return best_model, test_forecast[['yhat']], mape


def fit_prophet(DF, COL, changepoint_prior_scale, seasonality_prior_scale, yearly_seasonality, fourier_order):
    """
    Fit the Prophet model with right parameter

    Param:
    - DF:                   Dataframe
    - COL:                  Column
    - several parameters:   Parameter for model

    Return:
    - model
    """

    # Prepare data
    df_prophet = DF[[COL]].reset_index()
    df_prophet.rename(columns={'date': 'ds', COL: 'y'}, inplace=True)
    
    # Init model
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        yearly_seasonality=yearly_seasonality,
    )
    model.add_seasonality(name='yearly', period=365, fourier_order=fourier_order)
    
    # Train model
    model.fit(df_prophet)
    
    return model


def train_test_model(DF, CALLS_PER_DRIVER):
    """
    Train/test dem model (sby_planner) and visualize the results

    Params:
    - DF:                   Dataframe
    - COL:                  Column

    Return:
    - model
    """

    df_pred = pd.DataFrame(index=DF.index)

    # Define param_grid
    """param_grid = {
        'changepoint_prior_scale': [0.01],                    
        'seasonality_prior_scale': [75],                           
        'yearly_seasonality': [40],         
        'fourier_order': [20]
    }"""
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.25, 0.5, 0.75, 1.0],                    
        'seasonality_prior_scale': [10, 25, 50, 75, 100],                           
        'yearly_seasonality': [20, 30, 40, 50, 60],         
        'fourier_order': [20]
    }
    

    # Train model_1 (Prophet)
    # -----------------------
    # Generate model_1
    # Result: 0.01 / 75.0 / 40.0 / 20 (MAPE: 7.8%)
    model_1_calls, df_prediction_model_1, mape_prophet = crossvalidate_prophet(DF=DF, COL="calls", PARAM_GRID=param_grid)
    # Calc offset - where do the prediction start?
    offset = len(df_pred) - len(df_prediction_model_1)
    # Create new column for predicted values
    df_pred["pred_model_1"] = float("nan")
    # Append predicted values to the end of the df_pred
    df_pred.iloc[offset:, df_pred.columns.get_loc("pred_model_1")] = df_prediction_model_1["yhat"].values


    # Train model_2 (Theta)
    # ---------------------
    # Generate model_2
    # Result: 1.0 (MAPE: 12.37%)
    model_2_sick, df_prediction_model_2, mape_theta = optimize_theta(DF=DF, COL="n_sick_modified")
    # Calc offset - where do the prediction start?
    offset = len(df_pred) - len(df_prediction_model_2)
    # Create new column for predicted values
    df_pred["pred_model_2"] = float("nan")
    # Append predicted values to the end of the df_pred
    df_pred.iloc[offset:, df_pred.columns.get_loc("pred_model_2")] = df_prediction_model_2["n_sick_modified"].values


    # TEST THE MODEL
    # --------------
    # Train/Test Split
    df_train, df_test = dp.train_test_split(DF, 0.8)

    # Evaluate model
    df_pred["sum"] = (df_pred['pred_model_1'] * (1+mape_prophet/100) + df_pred['pred_model_2'] * (1+mape_theta/100))
    df_test["sum"] = df_test["calls"] + df_test["n_sick"]

    cf.validate(df_test["sum"].dropna(), df_pred["sum"].dropna())



    # Calculate pred_n_sby
    df_pred["pred_n_sby"] = (df_pred["pred_model_1"] * (1+mape_prophet/100) / CALLS_PER_DRIVER 
                             + df_pred["pred_model_2"] * (1+mape_theta/100) 
                             - DF["n_duty"])  

    # Cut values under 0
    df_pred.loc[df_pred["pred_n_sby"] < 0, "pred_n_sby"] = 0

    # Plot testdata and prediction
    cf.plot_combined_datasets(df_pred[["pred_n_sby"]], df_test[["sby_need"]], 
                            "2016-04-01", "2019-08-31", 
                            "Q",
                            COL_DF2="0.75",
                            LINESTYLE_DF2="-",
                            LINEWIDTH_DF1=1.5,
                            FIGSIZE=(16,4))
    

    # Find best offset and bias
    df_tmp = pd.DataFrame()
    for gain in np.arange(1, 10, 0.01):
        df_pred["pred_n_sby"] = (df_pred["pred_model_1"] * (1+mape_prophet*gain/100) / CALLS_PER_DRIVER 
                             + df_pred["pred_model_2"] * (1+mape_theta*gain/100) 
                             - DF["n_duty"])  
        # Cut values under 0
        df_pred.loc[df_pred["pred_n_sby"] < 0, "pred_n_sby"] = 0       
        # Get bias via difference
        bias = np.max(df_test["sby_need"] - df_pred["pred_n_sby"].dropna())
        # Add bias to data
        df_pred["pred_n_sby"] += bias
        # Calculate the total n_sby
        # Save the parameters
        df_tmp = pd.concat([df_tmp, pd.DataFrame({"gain": [gain], "bias": [bias], "sum_n_sby": [df_pred['pred_n_sby'].sum()]})], ignore_index=True)

    # Get best i with lowest n_sby
    min_sby_row = df_tmp.loc[df_tmp["sum_n_sby"].abs().idxmin()]
    best_gain = min_sby_row["gain"]
    best_bias = min_sby_row["bias"]

    print("\n\nGET BEST GAIN AND BIAS:\n-------------------------")
    print(f"GAIN:\t{best_gain:.3f}")
    print(f"BIAS:\t{best_bias:.3f}")
    print("-------------------------\n")


    # Reset pred_n_sby
    df_pred["pred_n_sby"] = (df_pred["pred_model_1"] * (1+mape_prophet/100) / CALLS_PER_DRIVER 
                             + df_pred["pred_model_2"] * (1+mape_theta/100) 
                             - DF["n_duty"])  
    # Cut values under 0
    df_pred.loc[df_pred["pred_n_sby"] < 0, "pred_n_sby"] = 0 


    # Calculate pred_n_sby_offset
    df_pred["pred_n_sby_offset"] = (df_pred["pred_model_1"] * (1+mape_prophet*best_gain/100) / CALLS_PER_DRIVER 
                             + df_pred["pred_model_2"] * (1+mape_theta*best_gain/100) 
                             - DF["n_duty"])
    # Cut values under 0
    df_pred.loc[df_pred["pred_n_sby_offset"] < 0, "pred_n_sby_offset"] = 0

    df_pred["pred_n_sby_offset"] += best_bias


    # Plot testdata and prediction
    cf.plot_combined_datasets(df_pred[["pred_n_sby_offset"]], df_pred[["pred_n_sby"]], 
                            "2016-04-01", "2019-08-31", 
                            "Q",
                            COL_DF2=gp._COLOR_,
                            LINESTYLE_DF2="--",
                            LINEWIDTH_DF1=1.5,
                            FIGSIZE=(16,4))
    
    # Plot testdata and prediction
    cf.plot_combined_datasets(df_pred[["pred_n_sby_offset"]], df_test[["sby_need"]], 
                            "2016-04-01", "2019-08-31", 
                            "Q",
                            COL_DF2="0.75",
                            LINESTYLE_DF2="-",
                            LINEWIDTH_DF1=1.5,
                            FIGSIZE=(16,4))
    

def predict_theta(DF, MODEL, PERIOD):
    """
    Forecast with Theta and viusalize (optional)

    Params:    
    - DF:       Dataframe to get the last index to append predicted values
    - COL:      Column
    - PERIOD:   Timeperiod to predict

    Return:
    - df_forecast[['yhat']]
    """

    last_date = DF.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PERIOD, freq='D').date   

    forecast = MODEL.predict(PERIOD)
    df_forecast = pd.DataFrame(np.nan, index=future_dates, columns=["yhat"])
    df_forecast["yhat"] = forecast.values()

    """# Visualisierung des Forecasts
    plt.rcdefaults()

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.plot(DF.index, DF["n_sick_modified"], label='Historical Data', color='black', linewidth=1)
    ax.plot(df_forecast.index, df_forecast['yhat'], label='Forecast', color=gp._COLOR_, linestyle='-', linewidth=1)

    ax.set_ylabel("Calls")
    ax.legend(loc="best", fontsize=gp._FONTSIZE_LEGEND_)
    ax.tick_params(labelsize=gp._FONTSIZE_TICK_)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)
    #ax.grid(True, axis='y', linestyle='--', color='gray', linewidth=0.6)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    plt.tight_layout(pad=2.0)
    plt.show()"""

    return df_forecast[['yhat']]


def predict_prophet(MODEL, PERIOD):
    """
    Forecast with Theta and viusalize (optional)

    Params:    
    - MODEL:    Model
    - PERIOD:   Timeperiod to predict

    Return:
    - df_forecast[['yhat']]
    """

    last_date = MODEL.history['ds'].max()
    future_dates = pd.DataFrame({'ds': pd.date_range(start=last_date + pd.Timedelta(days=1), periods=PERIOD, freq='D')})
    forecast = MODEL.predict(future_dates)

    """# Visualisierung des Forecasts
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    ax.plot(MODEL.history['ds'], MODEL.history['y'], label='Historical Data', color='black', linewidth=1)
    ax.plot(forecast['ds'], forecast['yhat'], label='Forecast', color=gp._COLOR_, linestyle='-', linewidth=1)

    ax.set_ylabel("Calls")
    ax.legend(loc="best", fontsize=gp._FONTSIZE_LEGEND_)
    ax.tick_params(labelsize=gp._FONTSIZE_TICK_)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)
    #ax.grid(True, axis='y', linestyle='--', color='gray', linewidth=0.6)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    plt.tight_layout(pad=2.0)
    plt.show()"""

    return forecast[['yhat']]


def forecast(DF, PERIOD, N_DUTY, CPD, MAPE_1, MAPE_2, GAIN, BIAS):
    """
    Forecast with Theta and viusalize (optional)

    Params:    
    - DF:       Dataframe
    - PERIOD:   Timeperiod to predict
    - N_DUTY:   Staff
    - CPD:      Calls_per_Driver (5)
    - MAPE_1:   MAPE_1 for Offset/Gain
    - MAPE_2:   MAPE_2 for Offset/Gain
    - GAIN:     Gain to avoid "dafted"
    - BIAS:     Bias to avoid "dafted"

    Return:
    - df_forecast[['yhat']]
    """
    # DF for forecasts
    df_forecast = pd.DataFrame(index=DF.index)
    df_forecast["forecast_calls"] = float("Nan")
    df_forecast["forecast_n_sick"] = float("Nan")


    # MODEL_1_CALLS
    # -------------
    model_1_calls = fit_prophet(DF=DF, 
                                COL="calls", 
                                changepoint_prior_scale=0.01, 
                                seasonality_prior_scale=75.0, 
                                yearly_seasonality=40.0, 
                                fourier_order=20.0)

    df_forecast_calls = predict_prophet(MODEL=model_1_calls, PERIOD=PERIOD)
    df_forecast_calls = df_forecast_calls.rename(columns={"yhat": "forecast_calls"})
    

    # MODEL_2_N_SICK
    # --------------
    model_2_n_sick = fit_theta(DF=DF,
                               COL="n_sick_modified",
                               THETA = 1.0)
    
    df_forecast_n_sick = predict_theta(DF=DF, MODEL=model_2_n_sick, PERIOD=PERIOD)
    df_forecast_n_sick = df_forecast_n_sick.rename(columns={"yhat": "forecast_n_sick"})

    # SET DATEINDEX
    # Get last date from forecast
    last_date = df_forecast.index[-1]
    # Create new indices
    new_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=len(df_forecast_calls))
    # Bring it to the forecast df
    df_forecast_calls.index = new_dates
    df_forecast_n_sick.index = new_dates

    # Append forecast
    df_append = pd.concat([df_forecast_calls, df_forecast_n_sick], axis=1)
    df_forecast = pd.concat([df_forecast, df_append], axis=0).dropna()

    # Calculate forecast for sby
    offset_m1 = 1 + MAPE_1 * GAIN / 100
    offset_m2 = 1 + MAPE_2 * GAIN / 100
    df_forecast["forecast_sby"] = df_forecast['forecast_calls'] / CPD * offset_m1 + df_forecast['forecast_n_sick'] * offset_m2 - N_DUTY + BIAS
    
    df_forecast = df_forecast.astype('int')

    return df_forecast[["forecast_sby"]]

