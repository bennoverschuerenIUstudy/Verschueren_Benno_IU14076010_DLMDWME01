import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore

import _020_src._global_parameters as gp
from _020_src._030_Deployment import custom_functions as cf



def load_prepare_data(PATH):
    """
    Load and prepare data:

    Param:
    PATH    Path of the data

    Return: Dataframe for the loaded data
    """
    # Load data and set column "date" as index
    df = pd.read_csv(PATH, parse_dates=True)

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

    return df


def load_ref_data(PATH):
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

    df_accidents = load_ref_data("_010_data/_010_Raw//Verkehrsunfallkalender_Daten_2023.csv")

    cf.plot_data(df_accidents[["Verunglueckte Kinder (unter 15 J)", "Unfaelle Fahrrad", "Unfaelle Motorrad/-roller"]], 
              "2022-01-01", "2023-12-31", 
              "Q", 
              (16,9))


def show_referencePopulation():
    """
    Shows the population from the reference 
    (https://www.statistik-berlin-brandenburg.de/datenportal/mats-bevoelkerungsstand)
    """

    df_population = load_ref_data("_010_data/_010_Raw//VÖ_Bevölkerungsstand_BBB.csv")
    
    # Plot the data
    num_columns = len(df_population.columns)
    # Create subplots
    fig, ax = plt.subplots(num_columns, 1, figsize=(16,3), sharex=True)
    ax.plot(df_population.index, df_population, label="Population", linewidth=1.25, color="black", zorder=100)
    ax.set_ylabel("Population")
    # Format tickvalues at yaxis 
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1e-6:.2f}M"))

    # Set the X-axis ticks to the exact index points (no intermediate steps)
    ax.set_xticks(df_population.index)  # Ensure X-axis shows only the exact index values
    ax.tick_params(labelsize=gp._FONTSIZE_TICK_)
    
    # Add vertical gridlines
    ax.grid(False)
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

    # Modify plot frame
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    # Show plot
    plt.tight_layout()
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


def detect_outliers_iqr(DF, THRESHOLD=3):
    """
    Detects outliers of a dataframe.

    Param:
    - DF:           Dataframe, which has to be investigate -> based on the zscore
    - THRESHOLD:    Threshold for the z_scores
    """
    z_scores = zscore(DF)
    return abs(z_scores) > THRESHOLD


def train_test_split(DF, RATIO):
    """
    Create train- and testsplit

    Param:
    - DF:           Dataframe, which has to be split
    - RATIO:        ratio of the train/test split

    Return:
    - df_train      Traindata
    - df_test       Testdata
    """
    train_size = int(RATIO * len(DF))
    df_train = DF.iloc[:train_size]
    df_test = DF.iloc[train_size:]

    return df_train, df_test






