import os
import pandas as pd
import json
import process_data
import eda
import outliers
import regression_model

# File Paths
weather_dir = r".\data\raw\weather_raw_data"
electricity_dir = r".\data\raw\electricity_raw_data"

def load_weather_data(folder):
    weather_data = []
    for file in os.listdir(folder):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, file))
            df["date"] = pd.to_datetime(df["date"], errors='coerce', utc=True)
            weather_data.append(df)
    weather_df = pd.concat(weather_data, ignore_index=True)
    weather_df = weather_df.rename(columns={"date": "timestamp", "temperature_2m": "temperature"})
    weather_df["timestamp"] = pd.to_datetime(weather_df["timestamp"], errors='coerce', utc=True)
    return weather_df

def load_electricity_data(folder):
    electricity_data = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), 'r') as f:
                try:
                    data = json.load(f)["response"]["data"]
                    electricity_data.extend(data)
                except KeyError:
                    print(f"Unexpected error, Skipping file")
    electricity_df = pd.DataFrame(electricity_data)
    electricity_df["period"] = pd.to_datetime(electricity_df["period"], errors='coerce')
    electricity_df["value"] = pd.to_numeric(electricity_df["value"], errors='coerce')
    electricity_df.rename(columns={"period": "timestamp", "value": "demand"}, inplace=True)
    electricity_df["timestamp"] = pd.to_datetime(electricity_df["timestamp"], errors='coerce', utc=True)
    return electricity_df

if __name__ == "__main__":
    weather_df = load_weather_data(weather_dir)
    electricity_df = load_electricity_data(electricity_dir)

    merged_data_df = pd.merge(electricity_df, weather_df, on="timestamp", how="inner")
    merged_data_df.sort_values(by="timestamp", inplace=True)
    merged_data_df.to_csv("merged_data.csv", index=False)

    processed_data_df = process_data.pre_process_data(merged_data_df)
    processed_data_df.to_csv("processed_data.csv", index=False)

    eda.perform_eda(processed_data_df)

    clean_df = outliers.detect_outliers(processed_data_df)
    clean_df.to_csv("cleaned_data.csv", index=False)

    model, predictions = regression_model.perform_regression_model(clean_df)