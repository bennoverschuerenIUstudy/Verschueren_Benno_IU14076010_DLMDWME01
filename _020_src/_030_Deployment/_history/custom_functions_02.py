import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from numpy import mean, sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.collections import PolyCollection, LineCollection
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb




# Global variables
_FONTSIZE_TICK_ = 6
_FONTSIZE_LEGEND_ = 8
_COLOR_ = "red"
_DOTSIZE_ = 50


def load_data(PATH):
    """
    Loads data

    Param:
    PATH    Path of the data

    Return: Dataframe for the loaded data
    """

    df = pd.read_csv(PATH, parse_dates=True, sep=';')

    # Rename the column and set the date as the index
    df.rename(columns={df.columns[0]: 'date'}, 
              inplace=True)
    df.set_index('date', 
                 inplace=True)
    
    # Ensure the index is datetime
    df.index = pd.to_datetime(df.index, format='%d.%m.%Y')

    return df


def show_referenceDataAccidents():
    """
    Shows statistics for accidents from the reference 
    https://service.destatis.de/DE/verkehrsunfallkalender/
    """

    df_accidents = load_data("_data/Verkehrsunfallkalender_Daten_2023.csv")

    plot_data(df_accidents[["Verunglueckte Kinder (unter 15 J)", "Unfaelle Fahrrad", "Unfaelle Motorrad/-roller"]], 
              "2022-01-01", "2023-12-31", 
              "Q", 
              (16,8))


def show_referencePopulation():
    """
    Shows the population from the reference 
    (https://www.statistik-berlin-brandenburg.de/datenportal/mats-bevoelkerungsstand)
    """

    df_population = load_data("_data/VÖ_Bevölkerungsstand_BBB.csv")
    
    # Plot the data
    num_columns = len(df_population.columns)
    # Create subplots
    fig, axes = plt.subplots(num_columns, 1, figsize=(16,4), sharex=True)
    axes.plot(df_population.index, df_population, label="Population", linewidth=1.25, color="black", zorder=100)
    axes.set_ylabel("Population")
    # Format tickvalues at yaxis 
    axes.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e-6:.2f}M"))

    # Set the X-axis ticks to the exact index points (no intermediate steps)
    axes.set_xticks(df_population.index)  # Ensure X-axis shows only the exact index values
    axes.tick_params(labelsize=_FONTSIZE_TICK_)
    
    # Add vertical gridlines
    axes.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

    # Modify plot frame
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)

    # Show plot
    plt.tight_layout()#pad=2.0)
    plt.show()


def check_data(DF):
    """
    Function to check data.
    - Missing data
    - Continuity of data

    Param:
    - DF:   Dataframe, which has to be checked

    Return:
    - Checksum
    """
    checksum=0

    print("\n\nCHECK DATA\n-----------------------------------------")

    # Check, if values are missing
    if DF.isnull().sum().sum() == 0:
        print("[DEBUG] Missing data:\t\tSUCCESS")
    else:
        print("[DEBUG] Missing data:\t\tFAIL")
        checksum+=1

    # Check continuity of the data
    diff = DF.index.to_series().diff()

    if (diff[1:] == pd.Timedelta(days=1)).all():
        print("[DEBUG] Continuity of data:\tSUCCESS")
    else:
        print("[DEBUG] Continuity of data:\tFAIL")
        checksum+=1

    print("-----------------------------------------\n")

    return checksum == 0


def plot_data(DF, START_DATE, END_DATE, freq='M', FIGSIZE=(16, 16)):
    """
    Function to plot a DataFrame with adjustable x-axis tick frequency.

    Param:
    - DF:               DataFrame with time-indexed data to be plotted.
    - START_DATE:       Start date of the plot.
    - END_DATE:         End date of the plot.
    - freq:             Frequency of x-axis ticks ('D', 'W', 'M', 'Q', 'Y').
                            'D' = Daily, 'W' = Weekly, 'M' = Monthly, 
                            'Q' = Quarterly, 'Y' = Yearly.
    - FIGSIZE           Size of the plot.
    """

    # Map frequency to matplotlib date locators
    freq_locators = {
        'D': mdates.DayLocator(),
        'W': mdates.WeekdayLocator(),
        'M': mdates.MonthLocator(),
        'Q': mdates.MonthLocator(bymonth=[1, 4, 7, 10]),
        'Y': mdates.YearLocator()
    }
    
    if freq not in freq_locators:
        raise ValueError("Invalid frequency. Choose from 'D', 'W', 'M', 'Q', 'Y'.")

    # How many columns
    num_columns = len(DF.columns)

    # Filter DataFrames based on the date range
    DF = DF.loc[START_DATE:END_DATE]

    # Create subplots
    fig, axes = plt.subplots(num_columns, 1, figsize=FIGSIZE, sharex=True)

    # Generate date range for x-axis ticks
    major_locator = freq_locators[freq]
    formatter = mdates.DateFormatter('%Y-%m-%d')

    # If the DataFrame has only one column
    if num_columns == 1:
        axes = [axes]

    # Each column in one plot
    for i, column in enumerate(DF.columns):
        axes[i].plot(DF.index, DF[column], 
                     label=column, 
                     linewidth=1.25, 
                     color="black", 
                     zorder=100)
        
        axes[i].set_ylabel(column)

        # Set major ticks and format
        axes[i].xaxis.set_major_locator(major_locator)
        axes[i].xaxis.set_major_formatter(formatter)
        axes[i].tick_params(labelsize=_FONTSIZE_TICK_)

        # Add vertical gridlines
        axes[i].grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

        # Modify plot frame
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    # Show plot
    plt.tight_layout(pad=2.0)
    plt.show()


def plot_combined_datasets(DF1, DF2, START_DATE, END_DATE, FREQ='M', LINEWIDTH_DF1 = 1.25, LINEWIDTH_DF2 = 1.25, LINESTYLE_DF2="--", COL_DF2="black", FIGSIZE=(16, 4)):
    """
    Function to plot two DataFrames on the same plot with adjustable x-axis tick frequency.

    Param:
    - DF1:          First DataFrame with time-indexed data to be plotted.
    - DF2:          Second DataFrame with time-indexed data to be plotted.
    - START_DATE:   Start date of the plot.
    - END_DATE:     End date of the plot.
    - FREQ:         Frequency of x-axis ticks ('D', 'W', 'M', 'Q', 'Y').
                            'D' = Daily, 'W' = Weekly, 'M' = Monthly, 
                            'Q' = Quarterly, 'Y' = Yearly.
    - FIGSIZE:      Size of the plot.
    """
    # Map frequency to matplotlib date locators
    freq_locators = {
        'D': mdates.DayLocator(),
        'W': mdates.WeekdayLocator(),
        'M': mdates.MonthLocator(),
        'Q': mdates.MonthLocator(bymonth=[1, 4, 7, 10]),
        'Y': mdates.YearLocator()
    }
    
    if FREQ not in freq_locators:
        raise ValueError("Invalid frequency. Choose from 'D', 'W', 'M', 'Q', 'Y'.")

    # Filter DataFrames based on the date range
    DF1 = DF1.loc[START_DATE:END_DATE]
    DF2 = DF2.loc[START_DATE:END_DATE]

    # Create the plot
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Generate date range for x-axis ticks
    major_locator = freq_locators[FREQ]
    formatter = mdates.DateFormatter('%Y-%m-%d')

    # Plot each column
    ax.plot(DF1.index, DF1, label=DF1.columns[0], linewidth=LINEWIDTH_DF1, color=_COLOR_, zorder=100)
    ax.plot(DF2.index, DF2, label=DF2.columns[0], linewidth=LINEWIDTH_DF2, color=COL_DF2, linestyle=LINESTYLE_DF2)
 
    # Set labels and legend
    ax.set_ylabel(f"Values ({DF1.columns[0]} & {DF2.columns[0]})")    
    ax.legend(loc="best", fontsize=_FONTSIZE_LEGEND_)

    # Set major ticks and format
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(labelsize=_FONTSIZE_TICK_)

    # Add gridlines
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

    # Modify plot frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Show plot
    plt.tight_layout(pad=2.0)
    plt.show()


def plot_outliers(DF, DETECT_OUTLIERS_FUNC, START_DATE, END_DATE, FIGSIZE=(16, 6)):
    """
    Plots a DataFrame with a single column, highlights outliers, and displays quarterly ticks.

    Param:
    - DF:                     DataFrame with only one column and a date index.
    - DETECT_OUTLIERS_FUNC:   Function for identifying outliers.
    - START_DATE:             Start date for quarterly axis ticks.
    - END_DATE:               End date for quarterly axis ticks.
    - FIGSIZE:                Size of the plot (default: (10, 6)).    
    """
    if DF.shape[1] != 1:
        raise ValueError("Der DataFrame muss genau eine Spalte enthalten.")
    
    # Get columnnames
    column = DF.columns[0]

    # Create start date quarterly
    quarter_starts = pd.date_range(start=START_DATE, end=END_DATE, freq="QS")

    # Plot
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Plot original data
    ax.plot(DF.index, DF[column], 
            label=column, 
            linewidth=1.25, 
            color="black", 
            zorder=100)

    # Identify outliers
    outliers_mask = DETECT_OUTLIERS_FUNC(DF[column])
    outliers = DF[outliers_mask]
    print("\n\nSHOW OUTLIER\n--------------------")
    print(outliers)
    print("--------------------\n")
    
    # Scatterplot for outliers
    ax.scatter(outliers.index, outliers[column], alpha=1, color=_COLOR_, s=_DOTSIZE_, label='Outliers', zorder=10)
    ax.scatter(outliers.index, outliers[column], alpha=0.15, edgecolors="none", color=_COLOR_, s=_DOTSIZE_*8, label='Outliers', zorder=10)

    ax.set_ylabel(column)

    # Ticks quarterly
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    ax.set_xticks(quarter_starts)
    ax.tick_params(labelsize=_FONTSIZE_TICK_)

    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

    # Modify frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

    return outliers_mask


def plot_metrics_comparison(DF1, DF2):
    """
    Plots a two DataFrames as barplot for comparision (yearly) and prints an overview.

    Param:
    - DF1:  Dataframe 1
    - DF2:  Dataframe 2
    """

    df_combined = pd.concat([DF1, DF2], axis=0)

    # Group the Dataframe by year and create the sum
    df_yearly_sum = df_combined.resample('YE').sum()
    df_yearly_sum = df_yearly_sum.astype(int)

    df_yearly_sum["diff"] = df_yearly_sum.iloc[:, 1] - df_yearly_sum.iloc[:, 0]
    df_yearly_sum["%"] = df_yearly_sum["diff"] / df_yearly_sum.iloc[:, 1] * 100
    median_diff = round(df_yearly_sum["%"].median(), 2)

    print("\n\nSHOW REQUIRED DUTY (BY CALLS) VS. ACTIVE_DRIVERS (BOOKED BY SYSTEM)\n----------------------------------------------------------------")
    print(df_yearly_sum)
    print("----------------------------------------------------------------")
    print(f"DIFFERENCE (MEDIAN):\t{median_diff} %")
    print("----------------------------------------------------------------\n")

    # Delete column
    df_yearly_sum.drop(columns=["diff", "%"], inplace=True)

    # Create barplot
    ax = df_yearly_sum.plot(kind="bar", figsize=(16, 4), width=0.15, stacked=False, color=['black', _COLOR_], zorder=100)

    # Format the x axis
    ax.set_xticklabels(df_yearly_sum.index.strftime('%Y-%m-%d')) 

    # Define axis labels
    ax.set_xlabel("")
    ax.set_ylabel("Counts")

    # Add legend
    plt.legend(fontsize=_FONTSIZE_LEGEND_)

    # rotate ticks
    plt.xticks(rotation=0)
    ax.tick_params(labelsize=_FONTSIZE_TICK_)

    ax.grid(True, axis='y', linestyle='--', color='gray', linewidth=0.6)

    # Modify frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Layout optimieren und anzeigen
    plt.tight_layout()
    plt.show()
    

def scatter_correlation(X, Y, METHOD='pearson', X_LABEL=None, Y_LABEL=None):
    """
    Performs a correlation analysis for two specific series and creates a scatter plot for these series without using Seaborn.

    Param:
    - X:        pd.Series - The first series for the analysis.
    - Y:        pd.Series - The second series for the analysis.
    - METHOD:   str - Correlation METHOD ('pearson', 'kendall', 'spearman'). Default: 'pearson'.
    - X_LABEL:  Customized label for the x axis
    - Y_LABEL:  Customized label for the y axis  

    Returns:    
    - The correlation value between the two series.
    """
    # Calculate correlation between both dataframes
    correlation = X.corr(Y, method=METHOD)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
        
    plt.scatter(X, Y, 
                color="black", 
                edgecolor='None', 
                s=_DOTSIZE_,
                zorder=100)


    # Linear regression to show the correlation
    a, d = np.polyfit(X, Y, 1) 

    # Calculate the regressionsline
    regression_line = a * X + d
        
    # Add the line to the plot
    plt.plot(X, regression_line, 
            color=_COLOR_, 
            label='Correlation',
            zorder=200)

    plt.xlabel(X_LABEL if X_LABEL else X.name)
    plt.ylabel(Y_LABEL if Y_LABEL else Y.name)
    plt.tick_params(labelsize=_FONTSIZE_TICK_)

    # Modify plot frame
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.grid(linestyle="--", color="gray", linewidth=0.6)
    plt.tight_layout()
    plt.show()

    return correlation


def get_correlation(X, Y, METHOD="pearson"):
    """
    Returns the correlation of two Dataframes.

    Param:
    - X:    Dataframe 1
    - Y:    Dataframe 2

    Return:
    - Correlationfactor
    """
    return X.corr(Y, method=METHOD)


def detect_outliers_iqr(DF, THRESHOLD=3):
    """
    Detects outliers of a dataframe.

    Param:
    - DF:           Dataframe, which has to be investigate -> based on the zscore
    - THRESHOLD:    Threshold for the z_scores
    """
    z_scores = zscore(DF)
    return abs(z_scores) > THRESHOLD


def linear_interpolation(Y1, Y2):
    """
    Creates a linear interpolated value between two values (neighbours)

    Param:
    - Y1:   Previous datapoint
    - Y2:   Next datapoint

    Return:
    - Interpolated value
    """

    return (Y2 + Y1)/2


def validate(DF, DF_PRED):    
    rmse = sqrt(mean_squared_error(DF, DF_PRED))
    percent_rmse = (rmse / DF.mean()) * 100
    correlation = get_correlation(DF_PRED, DF)
    print("\n\nVALIDATE\n------------------------------------")
    print(f"RMSE:\t\t{rmse:.3f} ({percent_rmse:.3f}%)")
    print(f"Correlation:\t{correlation:.3f}")
    print("------------------------------------\n")

    return rmse, correlation


def baseline_prediction(DF, COL, TRAIN_TEST_RATIO=0.8, FREQ="M"):
    """
    Creates a prediction based on additive composition via trend and seasonality.

    Param:
    - DF                Dataframe to predict
    - TRAIN_TEST_RATIO  Split of train- and testdata
    - FREQ:             Frequency of x-axis ticks ('D', 'W', 'M', 'Q', 'Y').
                        'D' = Daily, 'W' = Weekly, 'M' = Monthly, 
                        'Q' = Quarterly, 'Y' = Yearly.

    Return:
    - Dataframe of the prediction
    """

    # Map frequency to matplotlib date locators
    freq_locators = {
        'D': mdates.DayLocator(),
        'W': mdates.WeekdayLocator(),
        'M': mdates.MonthLocator(),
        'Q': mdates.MonthLocator(bymonth=[1, 4, 7, 10]),
        'Y': mdates.YearLocator()
    }
    
    if FREQ not in freq_locators:
        raise ValueError("Invalid frequency. Choose from 'D', 'W', 'M', 'Q', 'Y'.")

    # Create df for trainData and testData
    train_test_ratio = int(len(DF) * TRAIN_TEST_RATIO)
    df_train_data = DF.iloc[:train_test_ratio][[COL]]
    df_test_data = DF.iloc[train_test_ratio:][[COL]]

    # Definiere die Anzahl der Jahre, die für die Prognose verwendet werden sollen
    years = 2  # Beispiel: 3 Jahre
    days_per_year = 365

    # ** Calculate Seasonality **
    history = [x for x in df_train_data[COL]]   # Init historical data
    seasonal_predictions = list()       # list for the predictions

    for i in range(len(df_test_data)):
        # Get observations of the last timeperiod (years)
        obs = list()
        for y in range(1, years + 1):
            if len(history) >= y * days_per_year:
                obs.append(history[-(y * days_per_year)])
        # Add to historical data (for the next round)
        history.append(df_test_data.iloc[i])
        # Get the mean of the values and append to the predictions
        seasonal_predictions.append(mean(obs))

    # Build the final prediction (additive composition)
    final_predictions = np.array(seasonal_predictions) #+ (trend_predictions - b) / m
    df_predictions = pd.DataFrame(final_predictions, index=df_test_data.index, columns=[COL]).astype(int)

    validate(df_predictions[COL], df_test_data[COL])

    # Create the plot
    fig, ax = plt.subplots(figsize=(16,4))
    ax.plot(df_train_data.index, df_train_data, linewidth=1.25, label="Traindata", color="black")
    ax.plot(df_test_data.index, df_test_data, linewidth=1.25, label="Testdata", color="0.75")
    ax.plot(df_test_data.index, final_predictions, linewidth=1.25, label=f"Prediction", linestyle="-", color=_COLOR_)
    #ax.plot(df_test_data.index, trend_predictions, linewidth=1.25, label="Trend", linestyle="--", color="black")
    ax.set_ylabel("Calls")
    ax.legend(fontsize=_FONTSIZE_LEGEND_)
    ax.tick_params(labelsize=_FONTSIZE_TICK_)

    # Generate date range for x-axis ticks
    major_locator = freq_locators[FREQ]
    formatter = mdates.DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_major_locator(major_locator)

    # Modify plot frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

    # Layout optimieren und anzeigen
    plt.tight_layout()
    plt.show()

    return df_predictions


def plot_seasonal_decomposition(DF, COLUMN='calls', PERIOD=365):
    """
    Performs additive decomposition on a time series and plots the components.
    
    Param:
    - DF:           Dataframe to investigate
    - COLUMN:       The column name of the time series data.
    - PERIOD:       The seasonal period for decomposition.
    Return:
    - Result of the decomp
    """

    # Additive Composition via Decomposition
    # INFO: The method loses values at the beginning and end of the time series due to the moving average calculation.
    result = seasonal_decompose(DF[COLUMN], model="additive", period=PERIOD)
    
    # Plot Trend, Seasonality, and Residuals
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    
    axes[0].plot(DF.index, DF[COLUMN], label=COLUMN, color='black')
    axes[0].set_label(COLUMN)
    
    axes[1].plot(result.trend, label='Trend', color='black')
    axes[1].set_label("Trend")
    
    axes[2].plot(result.seasonal, label='Seasonality', color='black')
    axes[2].set_label("Seasonality")
    
    axes[3].plot(result.resid, label='Residuals', color='black')
    axes[3].set_label("Residuals")
    
    major_locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
    formatter = mdates.DateFormatter('%Y-%m-%d')
    
    for ax in axes:
        ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=_FONTSIZE_TICK_)
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.set_ylabel(ax.get_label())
    
    plt.tight_layout()
    plt.show()

    return result


def XGBoost(DF, COL, SPLIT=0.):
    df_pred = DF[[COL]].copy()
    df_pred.columns = ["y"]

    # 1. Get Features erstellen: Historische Werte + Zeitmerkmale
    df_pred['month'] = df_pred.index.month
    df_pred['quarter'] = df_pred['month'].apply(lambda x: (x-1)//3 + 1)
    df_pred['day_of_year'] = df_pred.index.dayofyear
    df_pred['week_of_year'] = df_pred.index.isocalendar().week
    df_pred['day_of_week'] = df_pred.index.dayofweek
    df_pred['rolling_7'] = df_pred['y'].rolling(window=7).mean()
    df_pred['rolling_30'] = df_pred['y'].rolling(window=30).mean()
    df_pred['rolling_365'] = df_pred['y'].rolling(window=365).mean()

    # Zyklische Umwandlung für Tag des Jahres
    df_pred['day_of_year_sin'] = np.sin(2 * np.pi * df_pred['day_of_year'] / 365.25)
    df_pred['day_of_year_cos'] = np.cos(2 * np.pi * df_pred['day_of_year'] / 365.25)

    # Zyklische Umwandlung für Wochentag
    df_pred['day_of_week_sin'] = np.sin(2 * np.pi * df_pred['day_of_week'] / 7)
    df_pred['day_of_week_cos'] = np.cos(2 * np.pi * df_pred['day_of_week'] / 7)

    # Lag-Features (Vergangene Werte als Input)
    lags = [1, 7, 30, 365]  # 1 Tag, 1 Woche, 1 Monat, 1 Jahr
    for lag in lags:
        df_pred[f'lag_{lag}'] = df_pred['y'].shift(lag)

    # 2️⃣ Daten splitten: Training & Test
    split_index = int(len(df_pred) * SPLIT)
    train, test = df_pred.iloc[:split_index], df_pred.iloc[split_index:]

    # Features und Zielvariable definieren
    X_train = train.drop(columns=["y"])
    y_train = train['y']
    X_test = test.drop(columns=["y"])
    y_test = test['y']

    # XGBoost Modell trainieren
    model = xgb.XGBRegressor(n_estimators=1000, 
                             learning_rate=0.05, 
                             objective='reg:squarederror',
                             max_depth=6,
                            )
    
    model.fit(X_train, y_train)

    # 4️⃣ Vorhersage für den Testzeitraum
    pred_xgb = model.predict(X_test)
    df_pred["pred_XGBoost"] = pd.DataFrame(pred_xgb, columns=['y'], index=test.index)

    return df_pred["pred_XGBoost"]


