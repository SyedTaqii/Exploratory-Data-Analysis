import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def test_mcar(df):
    df_miss = df.isnull().astype(int)
    chi2, p, _, _ = chi2_contingency(df_miss.corr())
    return p 

def missing_data_consistencies(df):
    # Calculate missing counts and percentages
    missing_counts = df.isnull().sum()
    missing_percentage = (missing_counts / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing Count': missing_counts,
        'Missing Percentage': missing_percentage
    })
    
    print("Missing Value Summary:")
    print(missing_df)

    #Determine missing value type
    p_missing_value = test_mcar(df)
    if p_missing_value > 0.05:
        print("Missing data is most likely MCAR")
    else:
        print("Missing data is likely MAR or MNAR")  

    #perform imputation
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':  # string type values or other objects
                df[col].fillna(df[col].mode()[0], inplace=True) #mode
            else: 
                df[col].fillna(df[col].median(), inplace=True)  #median

    print("Missing values handled") 

    return df 

def duplicate_and_inconsistencies(df):
    df = df.drop_duplicates() #remove duplicates

    for col in df.select_dtypes(include=['number']):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        print(f"Outliers detected in {col}: {len(outliers)} rows")
    
    return df

def engineer_feature(df):
    if 'timestamp' in df.columns:
        df.loc[:, 'timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        df.loc[:, 'year'] = df['timestamp'].dt.year
        df.loc[:, 'month'] = df['timestamp'].dt.month
        df.loc[:, 'day'] = df['timestamp'].dt.day
        df.loc[:, 'hour'] = df['timestamp'].dt.hour
        df.loc[:, 'day_of_week'] = df['timestamp'].dt.dayofweek
        df.loc[:, 'is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

        def get_season(month):
            if month in [12, 1, 2]:  
                return "Winter"
            elif month in [3, 4, 5]:  
                return "Spring"
            elif month in [6, 7, 8]:  
                return "Summer"
            else:  
                return "Autumn"

        df.loc[:, 'season'] = df['month'].apply(get_season)

    return df


def normalize_standardize(df, method="normalize"):
    numeric_cols = df.select_dtypes(include=['number']).columns
    if method == "normalize":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def pre_process_data(df):
    #data_conversions
    df["demand"] = pd.to_numeric(df["demand"], errors='coerce')
    df.sort_values(by="timestamp", inplace=True)

    df = missing_data_consistencies(df)
    df = duplicate_and_inconsistencies(df)
    df = engineer_feature(df)
    df = normalize_standardize(df, method="normalize") # Change to "standardize" if needed

    return df
