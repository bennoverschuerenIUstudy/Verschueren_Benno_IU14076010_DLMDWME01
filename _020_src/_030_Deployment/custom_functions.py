import pandas as pd
from numpy import mean, sqrt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.dates as mdates

import _020_src._global_parameters as gp


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
        axes[i].tick_params(labelsize=gp._FONTSIZE_TICK_)

        # Add vertical gridlines
        axes[i].grid(False)
        axes[i].grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)
        #axes[i].grid(True, axis='y', linestyle='--', color='gray', linewidth=0.6)

        # Modify plot frame
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_visible(True)
        axes[i].spines['left'].set_visible(True)

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
    plt.rcdefaults()
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Generate date range for x-axis ticks
    major_locator = freq_locators[FREQ]
    formatter = mdates.DateFormatter('%Y-%m-%d')

    # Plot each column
    ax.plot(DF1.index, DF1, label=DF1.columns[0], linewidth=LINEWIDTH_DF1, color=gp._COLOR_, zorder=100)
    ax.plot(DF2.index, DF2, label=DF2.columns[0], linewidth=LINEWIDTH_DF2, color=COL_DF2, linestyle=LINESTYLE_DF2)
 
    # Set labels and legend
    ax.set_ylabel(f"{DF1.columns[0]} & {DF2.columns[0]}")    
    ax.legend(loc="best", fontsize=gp._FONTSIZE_LEGEND_)

    # Set major ticks and format
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(labelsize=gp._FONTSIZE_TICK_)

    # Add gridlines
    ax.grid(False)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

    # Modify plot frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

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
    quarter_starts = pd.date_range(start=START_DATE, end=END_DATE, freq="Q")


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
    ax.scatter(outliers.index, outliers[column], alpha=1, color=gp._COLOR_, s=gp._DOTSIZE_, label='Outliers', zorder=10)
    ax.scatter(outliers.index, outliers[column], alpha=0.15, edgecolors="none", color=gp._COLOR_, s=gp._DOTSIZE_*8, label='Outliers', zorder=10)

    ax.set_ylabel(column)

    # Ticks quarterly
    # Generate date range for x-axis ticks
    major_locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
    formatter = mdates.DateFormatter('%Y-%m-%d')

    # Set major ticks and format
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(labelsize=gp._FONTSIZE_TICK_)

    # Set grid
    ax.grid(False)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)
    
    # Modify frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)

    
    plt.tight_layout()
    plt.show()

    return outliers_mask
    

def scatter_correlation(X, Y, METHOD='pearson', X_LABEL=None, Y_LABEL=None, FIGSIZE=(4, 4)):
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
    plt.figure(figsize=FIGSIZE)
        
    plt.scatter(X, Y, 
                color="black", 
                edgecolor='None', 
                s=gp._DOTSIZE_,
                zorder=100)


    # Linear regression to show the correlation
    a, d = np.polyfit(X, Y, 1) 

    # Calculate the regressionsline
    regression_line = a * X + d
        
    # Add the line to the plot
    plt.plot(X, regression_line, 
            color=gp._COLOR_, 
            label='Correlation',
            zorder=200)

    plt.xlabel(X_LABEL if X_LABEL else X.name)
    plt.ylabel(Y_LABEL if Y_LABEL else Y.name)
    plt.tick_params(labelsize=gp._FONTSIZE_TICK_)

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
    mape = mean_absolute_percentage_error(DF, DF_PRED)
    correlation = get_correlation(DF_PRED, DF)

    print("\n\nVALIDATE\n------------------------------------")
    print(f"RMSE:\t\t{rmse:.3f} ({percent_rmse:.3f} %)")
    print(f"MAPE:\t\t{mape*100:.3f} %")
    print(f"Correlation:\t{correlation:.3f}")
    print("------------------------------------\n")

    return rmse, correlation

