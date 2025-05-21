import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

def load_data():
    base_path = os.path.join("data")
    train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
    stores_df = pd.read_csv(os.path.join(base_path, "stores.csv"))
    features_df = pd.read_csv(os.path.join(base_path, "features.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "test.csv"))
    return train_df, stores_df, features_df, test_df

def preprocess_data(train_df, stores_df, features_df):
    # Merge datasets
    train_df = train_df.merge(stores_df, on='Store', how='left')
    train_df = train_df.merge(features_df, on=['Store', 'Date'], how='left')

    # Fill missing markdowns with 0
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    train_df[markdown_cols] = train_df[markdown_cols].fillna(0)

    # Convert date to datetime and extract features
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    train_df['Year'] = train_df['Date'].dt.year
    train_df['Month'] = train_df['Date'].dt.month
    train_df['Week'] = train_df['Date'].dt.isocalendar().week
    train_df['DayOfWeek'] = train_df['Date'].dt.day_of_week

    # Encode categorical variables
    train_df['Type'] = train_df['Type'].map({'A': 0, 'B': 1, 'C': 2})
    train_df['IsHoliday_x'] = train_df['IsHoliday_x'].astype(int)
    train_df['IsHoliday_y'] = train_df['IsHoliday_y'].astype(int)

    return train_df

def train_model(train_df):
    features = ['Store', 'Dept', 'Size', 'Type', 'Year', 'Month', 'Week', 'DayOfWeek', 
                'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 
                'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment', 'IsHoliday_x', 'IsHoliday_y']
    
    X = train_df[features]
    y = train_df['Weekly_Sales']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize selected features
    scaler = StandardScaler()
    scale_cols = ['Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
    X_test[scale_cols] = scaler.transform(X_test[scale_cols])

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

if __name__ == "__main__":
    train_df, stores_df, features_df, test_df = load_data()
    train_df = preprocess_data(train_df, stores_df, features_df)
    train_model(train_df)

