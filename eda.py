import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

def perform_stats_analysis(df):
    """Compute basic statistics (mean, median, std, etc.) for each numerical column."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns  
    stats_df = pd.DataFrame(columns=["Mean", "Median", "Standard Deviation", "Variance", 
                                     "Skewness", "Kurtosis", "Min", "Max"])

    for col in numeric_cols:
        stats_df.loc[col] = {
            "Mean": df[col].mean(),
            "Median": df[col].median(),
            "Standard Deviation": df[col].std(),
            "Variance": df[col].var(),
            "Skewness": df[col].skew(),
            "Kurtosis": stats.kurtosis(df[col], fisher=True),
            "Min": df[col].min(),
            "Max": df[col].max()
        }
    
    print("\nStatistical Summary of Numeric Features")
    print(stats_df)

def plot_time_series(df):
    """Plot electricity demand over time with a line plot."""

    timestamp_col = "timestamp"
    demand_col = "demand"
    df.loc[:, timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.sort_values(by=timestamp_col)

    plt.figure(figsize=(12, 6))
    plt.plot(df[timestamp_col], df[demand_col], label="Electricity Demand", color='blue')
    plt.title("Electricity Demand Over Time", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Electricity Demand", fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.show()

def perform_univariate_analysis(df):
    """Generate histograms, boxplots, and density plots for numerical features."""

    df.loc[:, "timestamp"] = pd.to_datetime(df["timestamp"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for column in numeric_cols:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 3, 1)
        sns.histplot(df[column], kde=True)
        plt.title(f'Histogram of {column}')

        plt.subplot(1, 3, 2)
        sns.boxplot(y=df[column])
        plt.title(f"Boxplot of {column}")

        plt.subplot(1, 3, 3)
        sns.kdeplot(df[column], fill=True)
        plt.title(f"Density Plot of {column}")

        plt.show()

def perform_correlation_analysis(df):
    """Compute and visualize correlation matrix for numerical features."""

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.show()

def perform_advance_time_series_analysis(df):
    """Decompose the time series and perform a stationarity test."""

    timestamp_col = "timestamp"
    demand_col = "demand"
    df.loc[:, timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)

    decomposition = seasonal_decompose(df[demand_col], model='additive', period=365)
    decomposition.plot()
    plt.show()
    perform_stationarity_test(df, timestamp_col, demand_col)

def perform_stationarity_test(df, timestamp_col, demand_col):
    """Conduct Augmented Dickey-Fuller test to check for stationarity."""
    
    result = adfuller(df[demand_col].dropna())
    print("\nAugmented Dickey-Fuller Test Results")
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    print(f"Critical Values: {result[4]}")
    
    if result[1] < 0.05:
        print("The time series is stationary (rejecting null hypothesis).")
    else:
        print("The time series is non-stationary (fail to reject null hypothesis).")
    df.reset_index(inplace=True)

def perform_eda(df):

    perform_stats_analysis(df)
    plot_time_series(df)
    perform_univariate_analysis(df)
    perform_correlation_analysis(df)
    perform_advance_time_series_analysis(df)
