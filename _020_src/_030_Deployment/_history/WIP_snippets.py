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

    # Calculate Trend
    train_x = np.arange(len(df_train_data[COL])).reshape(-1, 1)  # Umwandlung in 2D-Array (n_samples, n_features)
    test_x = np.arange(len(df_train_data[COL]), len(df_train_data[COL]) + len(df_test_data)).reshape(-1, 1)

    # Erstelle das Modell für die lineare Regression
    #model_linReg = LinearRegression().fit(train_x, df_train_data[COL])

    # Calculate trend via linearRegression (y_t = mx + b)
    #trend_predictions = model_linReg.predict(test_x)
    # Get parameters of the linearRegression
    #b = model_linReg.intercept_     # Bias
    #m = model_linReg.coef_[0]       # Slope

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

    validate(df_test_data[COL], df_predictions[COL])

    # Create scatter plot
    scatter_correlation(df_test_data[COL], df_predictions[COL], X_LABEL="Prediction", Y_LABEL="Testdata")

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







def plot_acf_pacf(DF, LAGS=400):
    """
    Plots the Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
    for a given time series.
    
    Param:
    - DF:               Dataframe
    - LAGS:             Number of lags to display in the plots.
    """

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # ACF plot
    plot_acf(DF.diff().dropna(), 
            lags=LAGS, 
            ax=axes[0], 
            zero=False, 
            color="black",
            vlines_kwargs={'colors': _COLOR_}, 
            title="")
    axes[0].set_ylabel('ACF')

    # PACF plot
    plot_pacf(DF.diff().dropna(), 
              lags=LAGS, ax=axes[1], 
              method='ywm', 
              zero=False, 
              color="black", 
              vlines_kwargs={'colors': _COLOR_}, 
              title="")
    axes[1].set_ylabel('PACF')

    for ax in axes:
            ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.tick_params(labelsize=_FONTSIZE_TICK_)

            for item in ax.collections:
                # change the color of the confidence interval 
                if type(item) == PolyCollection:
                    item.set_facecolor(_COLOR_)
                # change the color of the vertical lines
                if type(item) == LineCollection:
                    item.set_color(_COLOR_)

    







    plt.tight_layout(pad=2)
    plt.show()


def SARIMA(DF, COLUMN, TRAIN, TEST, p=1, d=1, q=1, P=1, D=1, Q=1, S=365):
    """
    Trainiert ein SARIMA-Modell und zeigt die Vorhersage zusammen mit den tatsächlichen Werten an.
    
    Parameters:
    df (DataFrame): Das Zeitreihendaten-Frame mit einer DatetimeIndex und Zielspalte.
    target_column (str): Der Name der Spalte, die vorhergesagt werden soll.
    p, d, q (int): ARIMA-Parameter.
    P, D, Q, S (int): Saisonale Parameter (für saisonale Zeitreihen).
    """
    


    print("Training des SARIMA-Modells läuft...")
    from sklearn.preprocessing import StandardScaler
    # SARIMA-Modell anpassen
    model = SARIMAX(
        TRAIN[COLUMN],
        order=(p, d, q),
        seasonal_order=(P, D, Q, S),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    model_fit = model.fit(progress=True, disp=0, maxiter=50, method='powell')
    
    # Vorhersage für den Testdatensatz
    predictions = model_fit.forecast(steps=len(TEST))
    
    # Convert to dataframe
    df_predictions = pd.DataFrame(predictions, index=TEST.index, columns=["calls"])
    
    # Erstelle das Plot
    major_locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
    formatter = mdates.DateFormatter('%Y-%m-%d')

    fig, ax = plt.subplots(figsize=(16, 4))

    ax.plot(TRAIN.index, TRAIN[COLUMN], label="Trainingdata", color="black")
    ax.plot(TEST.index, TEST[COLUMN], label="Testdata", color="0.75")
    ax.plot(TEST.index, predictions, label="Prediction", color=_COLOR_)
    
    # Setze Labels und Legende
    ax.set_ylabel(COLUMN)    
    ax.legend(loc="best", fontsize=8)

    # Setze Major-Ticks und Formatierung
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.tick_params(labelsize=6)

    # Füge Gitterlinien hinzu
    ax.grid(True, axis='x', linestyle='--', color='gray', linewidth=0.6)

    # Modifiziere den Rahmen des Plots
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Zeige den Plot
    plt.tight_layout()
    plt.show()

    return predictions




def check_stationarity(DF, SIGNIFICANCE_LEVEL=0.05):
    """
    Checks the stationarity of a time series using the Augmented Dickey-Fuller (ADF) test.
    
    Parameters:
    - DF:                   The time series to check (e.g., Pandas Series).
    - SIGNIFICANCE_LEVEL:   Significance level for the test (default is 0.05).
    
    Prints:
    - Test statistic, p-value, critical values, and whether the time series is stationary.
    
    Returns:
    - True if the time series is stationary, otherwise False.
    """
    # Perform the Augmented Dickey-Fuller (ADF) test
    result = adfuller(DF, autolag='AIC')
    
    # Extract test results
    test_statistic = result[0]
    p_value = result[1]
    is_stationary = p_value < SIGNIFICANCE_LEVEL  # Time series is stationary if p-value < significance level
    
    # Print the results
    print("\n\nCHECK STATIONARITY\n---------------------------------")
    print(f"Test Statistic:\t\t{test_statistic:.4f}")
    print(f"p-value:\t\t{p_value:.4f}")
    print(f"Significance Level:\t{SIGNIFICANCE_LEVEL}")
    print(f"\nStationary:\t\t{'TRUE' if is_stationary else 'FALSE'}")
    print("---------------------------------\n")

    
    # Return True if stationary, otherwise False
    return is_stationary
