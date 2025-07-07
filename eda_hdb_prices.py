import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def perform_eda(file_path):
    """
    Performs Exploratory Data Analysis on the HDB resale price dataset.

    Args:
        file_path (str): The absolute path to the CSV file.
    """
    # --- 1. Load Data ---
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded data from {file_path}")
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return

    # --- 2. Initial Inspection ---
    print("\n--- Initial Data Inspection ---")
    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Information:")
    df.info()

    print("\nDescriptive Statistics for Numerical Columns:")
    print(df.describe())

    # --- 3. Data Cleaning and Preprocessing ---
    print("\n--- Data Cleaning and Preprocessing ---")

    # Check for duplicates
    print(f"\nNumber of duplicate rows: {df.duplicated().sum()}")
    df.drop_duplicates(inplace=True)
    print(f"Dropped duplicates. New shape: {df.shape}")

    # Check for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    # No missing values found, which is great.

    # Convert 'month' to datetime objects
    df['month'] = pd.to_datetime(df['month'], format='%Y-%m')

    # Convert 'remaining_lease' from string to a numerical value (years)
    def parse_lease_to_years(lease_str):
        """Converts lease string e.g., '61 years 04 months' to a float."""
        if pd.isna(lease_str):
            return None
        
        years = 0
        months = 0
        
        # Using regex to find years and months
        year_match = re.search(r'(\d+)\s+years?', lease_str)
        month_match = re.search(r'(\d+)\s+months?', lease_str)
        
        if year_match:
            years = int(year_match.group(1))
        if month_match:
            months = int(month_match.group(1))
            
        return years + months / 12

    df['remaining_lease_years'] = df['remaining_lease'].apply(parse_lease_to_years)
    print("\nConverted 'remaining_lease' to 'remaining_lease_years' (numerical).")

    # Convert 'storey_range' to a numerical value (mid-point of the range)
    df['storey_mid'] = df['storey_range'].apply(lambda x: (int(x.split(' TO ')[0]) + int(x.split(' TO ')[1])) / 2)
    print("Created 'storey_mid' from 'storey_range'.")

    print("\nCleaned DataFrame Info:")
    df.info()

    # --- 4. Univariate Analysis (Visualizing Single Variables) ---
    print("\n--- Generating Visualizations ---")
    sns.set_style("whitegrid")

    # Distribution of Resale Price
    plt.figure(figsize=(12, 6))
    sns.histplot(df['resale_price'], kde=True, bins=50)
    plt.title('Distribution of Resale Prices')
    plt.xlabel('Resale Price (SGD)')
    plt.ylabel('Frequency')
    plt.ticklabel_format(style='plain', axis='x')
    plt.show()

    # Distribution of Floor Area
    plt.figure(figsize=(12, 6))
    sns.histplot(df['floor_area_sqm'], kde=True, bins=40)
    plt.title('Distribution of Floor Area (sqm)')
    plt.xlabel('Floor Area (sqm)')
    plt.ylabel('Frequency')
    plt.show()

    # Count of transactions by Town
    plt.figure(figsize=(12, 10))
    sns.countplot(y='town', data=df, order=df['town'].value_counts().index)
    plt.title('Number of Resale Transactions by Town')
    plt.xlabel('Number of Transactions')
    plt.ylabel('Town')
    plt.show()

    # Count by Flat Type
    plt.figure(figsize=(10, 6))
    sns.countplot(x='flat_type', data=df, order=df['flat_type'].value_counts().index)
    plt.title('Number of Resale Transactions by Flat Type')
    plt.xlabel('Flat Type')
    plt.ylabel('Number of Transactions')
    plt.xticks(rotation=45)
    plt.show()

    # --- 5. Bivariate Analysis (Visualizing Relationships) ---

    # Resale Price vs. Floor Area
    plt.figure(figsize=(12, 7))
    sns.scatterplot(x='floor_area_sqm', y='resale_price', data=df, alpha=0.5, s=15)
    plt.title('Resale Price vs. Floor Area')
    plt.xlabel('Floor Area (sqm)')
    plt.ylabel('Resale Price (SGD)')
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()

    # Resale Price by Flat Type
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='flat_type', y='resale_price', data=df, order=sorted(df['flat_type'].unique()))
    plt.title('Resale Price by Flat Type')
    plt.xlabel('Flat Type')
    plt.ylabel('Resale Price (SGD)')
    plt.ticklabel_format(style='plain', axis='y')
    plt.xticks(rotation=45)
    plt.show()

    # Average Resale Price by Town
    plt.figure(figsize=(12, 10))
    avg_price_by_town = df.groupby('town')['resale_price'].mean().sort_values(ascending=False)
    sns.barplot(y=avg_price_by_town.index, x=avg_price_by_town.values)
    plt.title('Average Resale Price by Town')
    plt.xlabel('Average Resale Price (SGD)')
    plt.ylabel('Town')
    plt.show()

    # --- 6. Time Series Analysis ---

    # Average Resale Price Over Time
    plt.figure(figsize=(15, 7))
    avg_price_over_time = df.groupby('month')['resale_price'].mean()
    avg_price_over_time.plot(kind='line')
    plt.title('Average HDB Resale Price Over Time (2017 onwards)')
    plt.xlabel('Date')
    plt.ylabel('Average Resale Price (SGD)')
    plt.grid(True)
    plt.show()

    # --- 7. Correlation Analysis ---
    
    # Correlation Heatmap for numerical features
    plt.figure(figsize=(10, 8))
    numerical_cols = ['floor_area_sqm', 'lease_commence_date', 'remaining_lease_years', 'storey_mid', 'resale_price']
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()


if __name__ == '__main__':
    # IMPORTANT: Replace this with the actual path to your CSV file.
    csv_file_path = r'c:\Users\lifel\Downloads\HDBprices\ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv'
    perform_eda(csv_file_path)

