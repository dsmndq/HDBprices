import pandas as pd
import numpy as np
import re
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def parse_lease_to_years(lease_str):
    """Converts lease string e.g., '61 years 04 months' to a float."""
    if pd.isna(lease_str):
        return None
    
    years = 0
    months = 0
    
    year_match = re.search(r'(\d+)\s+years?', lease_str)
    month_match = re.search(r'(\d+)\s+months?', lease_str)
    
    if year_match:
        years = int(year_match.group(1))
    if month_match:
        months = int(month_match.group(1))
        
    return years + months / 12


def load_and_preprocess_data(file_path):
    """
    Loads and preprocesses the HDB resale data for machine learning.

    Args:
        file_path (str): The absolute path to the CSV file.

    Returns:
        pandas.DataFrame: A cleaned and preprocessed DataFrame ready for modeling.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return None

    # --- Data Cleaning (from EDA) ---
    df.drop_duplicates(inplace=True)

    # --- Feature Engineering ---
    # Convert 'month' to datetime and extract year/month features
    df['month_dt'] = pd.to_datetime(df['month'], format='%Y-%m')
    df['year'] = df['month_dt'].dt.year
    df['month'] = df['month_dt'].dt.month

    # Convert 'remaining_lease' to a numerical feature
    df['remaining_lease_years'] = df['remaining_lease'].apply(parse_lease_to_years)

    # Convert 'storey_range' to a numerical feature (mid-point)
    df['storey_mid'] = df['storey_range'].apply(lambda x: (int(x.split(' TO ')[0]) + int(x.split(' TO ')[1])) / 2)

    # --- Feature Selection and Encoding ---
    # Drop original, redundant, or high-cardinality string columns
    df = df.drop([
        'month_dt', 'remaining_lease', 'storey_range', 'block',
        'street_name'
    ], axis=1)

    # One-Hot Encode categorical features
    # These are nominal categories where there is no intrinsic order.
    categorical_cols = ['town', 'flat_type', 'flat_model']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("Data preprocessing and feature engineering complete.")
    return df


def train_and_evaluate_model(df):
    """
    Trains a LightGBM model and evaluates its performance.

    Args:
        df (pandas.DataFrame): The preprocessed DataFrame.
    """
    # --- 1. Define Features (X) and Target (y) ---
    X = df.drop('resale_price', axis=1)
    y = df['resale_price']

    # --- 2. Split Data into Training and Testing Sets ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Data split into training ({X_train.shape[0]} rows) and testing ({X_test.shape[0]} rows) sets.")

    # --- 3. Initialize and Train the Model ---
    print("\nTraining LightGBM Regressor model...")
    lgbm = lgb.LGBMRegressor(random_state=42)
    lgbm.fit(X_train, y_train)
    print("Model training complete.")

    # --- 4. Make Predictions and Evaluate ---
    print("\n--- Model Evaluation ---")
    y_pred = lgbm.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
    print(f"R-squared (R²): {r2:.4f}")

    # --- 5. Visualize Results ---
    # Feature Importance Plot
    plt.figure(figsize=(12, 10))
    lgb.plot_importance(lgbm, max_num_features=20, height=0.8, title='Top 20 Feature Importances')
    plt.show()

    # Actual vs. Predicted Plot
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.title('Actual vs. Predicted Resale Prices')
    plt.xlabel('Actual Price (SGD)')
    plt.ylabel('Predicted Price (SGD)')
    plt.ticklabel_format(style='plain')
    plt.show()


if __name__ == '__main__':
    # IMPORTANT: Ensure this path is correct.
    csv_file_path = r'c:\Users\lifel\Downloads\HDBprices\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv'
    
    processed_df = load_and_preprocess_data(csv_file_path)
    
    if processed_df is not None:
        train_and_evaluate_model(processed_df)
