import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(df, target_col, features_cols):
    X = df[features_cols]
    y = df[target_col]
    
    # Split into Training & Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model using MSE, RMSE, and R2 score
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R2 Score: {r2}')
    
    return model, y_test, y_pred

def plot_actual_and_predicted(y_test, y_pred):
    # Plotting actual and predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--') 
    plt.title('Actual vs Predicted Electricity Demand')
    plt.xlabel('Actual Demand')
    plt.ylabel('Predicted Demand')
    plt.show()

def residual_analysis(y_test, y_pred):
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Plotting residuals
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, color='blue', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')  # Horizontal line at 0
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

def perform_regression_model(df):
    timestamp_col = "timestamp"
    target_col = "demand"

    feature_cols = [ "temperature", "year", "month", "day", "hour", "day_of_week", "is_weekend"]

    model, y_test, y_pred = evaluate_model(df, target_col, feature_cols)

    plot_actual_and_predicted(y_test, y_pred)

    residual_analysis(y_test, y_pred)

    return model, y_pred