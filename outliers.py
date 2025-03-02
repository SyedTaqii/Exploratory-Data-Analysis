import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore


# IQR-based Outlier Detection
def detect_outliers_iqr(df):
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
    return outliers

# Z-score-based Outlier Detection
def detect_outliers_zscore(df, threshold=3):
    outliers = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        z_scores = zscore(df[col])
        outliers[col] = df[np.abs(z_scores) > threshold].index
    return outliers

def plot_data_before_after(df, df_cleaned_iqr, df_cleaned_zscore):
    # Plot original data
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 3, 1)
    sns.boxplot(data=df)
    plt.title('Original Data')

    # After modification (cleaned data)
    plt.subplot(1, 3, 2)
    sns.boxplot(data=df_cleaned_iqr)
    plt.title('After Handling IQR Outliers')

    plt.subplot(1, 3, 3)
    sns.boxplot(data=df_cleaned_zscore)
    plt.title('After Handling Z-Score Outliers')

    plt.show()

def handle_outliers(df, outliers, strategy="remove", cap_value=None):
    df_copy = df.copy()
    for col, indices in outliers.items():
        if strategy == "remove":
            df_copy = df_copy.drop(index=indices, errors='ignore')
        elif strategy == "cap":
            lower_limit = df_copy[col].quantile(0.05) if cap_value is None else cap_value
            upper_limit = df_copy[col].quantile(0.95) if cap_value is None else cap_value
            df_copy[col] = np.clip(df_copy[col], lower_limit, upper_limit)
        elif strategy == "transform":
            df_copy[col] = np.log1p(df_copy[col])  # Log transformation
    return df_copy

def detect_outliers(df):
    outliers_iqr = detect_outliers_iqr(df)
    outliers_zscore = detect_outliers_zscore(df)

    clenaed_df = df_iqr_removed = handle_outliers(df, outliers_iqr, strategy="remove")
    clenaed_df = df_zscore_transformed = handle_outliers(df, outliers_zscore, strategy="transform")

    plot_data_before_after(df, df_iqr_removed, df_zscore_transformed)
    return clenaed_df