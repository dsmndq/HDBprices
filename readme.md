# HDB Resale Price Prediction Project

This project aims to predict the resale prices of HDB flats in Singapore using machine learning. It includes scripts for Exploratory Data Analysis (EDA) and for training a predictive model using LightGBM.

Read about the [details and findings here](https://medium.com/@desmond_57481/decoding-the-hdb-property-market-using-machine-learning-to-explain-and-estimate-hdb-resale-prices-7d43b9d5d6e2)

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [1. Exploratory Data Analysis (EDA)](#1-exploratory-data-analysis-eda)
  - [2. Data Preprocessing & Feature Engineering](#2-data-preprocessing--feature-engineering)
  - [3. Modeling](#3-modeling)
- [Results](#results)
  - [Model Performance](#model-performance)
  - [Feature Importance](#feature-importance)
  - [Actual vs. Predicted Prices](#actual-vs-predicted-prices)
- [File Structure](#file-structure)
- [How to Run](#how-to-run)
  - [Prerequisites](#prerequisites)
  - [Running the Scripts](#running-the-scripts)
- [Future Work](#future-work)

## Project Overview

The primary goal is to build a regression model that accurately predicts the `resale_price` of an HDB flat based on its physical attributes, location, and transaction date. This can be valuable for buyers, sellers, and real estate agents to understand market pricing.

## Dataset

The dataset used is **"Resale flat prices based on registration date from Jan 2017 onwards"**, sourced from the Singapore government's open data portal, data.gov.sg.

## Methodology

The project is divided into two main parts: analysis and modeling.

### 1. Exploratory Data Analysis (EDA)

The `eda_hdb_prices.py` script performs a thorough EDA to understand the dataset's characteristics. Key steps include:
- Visualizing the distribution of key numerical features like `resale_price` and `floor_area_sqm`.
- Analyzing the number of transactions by `town` and `flat_type`.
- Plotting relationships between variables, such as `resale_price` vs. `floor_area_sqm`.
- Examining price trends over time.

### 2. Data Preprocessing & Feature Engineering

The `load_and_preprocess_data` function in the training script (`train_hdb_model_lightbgm.py`) performs the following crucial steps:
- **Duplicate Removal**: Removes any duplicate rows to ensure data quality.
- **Feature Creation**:
  - `remaining_lease_years`: Converts the lease string (e.g., '90 years 5 months') into a single numerical value.
  - `storey_mid`: Converts the storey range (e.g., '10 TO 12') into its numerical midpoint (11).
  - `year` & `month`: Extracts the year and month from the transaction date to capture time-based patterns.
- **Feature Selection**: Drops high-cardinality or redundant columns like `block` and `street_name`.
- **One-Hot Encoding**: Converts categorical features (`town`, `flat_type`, `flat_model`) into a numerical format that the model can process.

### 3. Modeling

The `train_hdb_model_lightbgm.py` script trains a **LightGBM Regressor**, a powerful and efficient gradient boosting model. The data is split into an 80% training set and a 20% testing set to evaluate the model's performance on unseen data.

## Results

The LightGBM model demonstrates excellent predictive performance on the test set.

### Model Performance

The evaluation metrics indicate a highly accurate model:

*   **R-squared (R²)**: `0.9781` (The model explains ~97.8% of the variance in resale prices)
*   **Mean Absolute Error (MAE)**: `$17,854.21` (On average, the model's prediction is off by about $17,800)
*   **Root Mean Squared Error (RMSE)**: `$24,315.98`

*(Note: These are sample results from a typical run of the script.)*

### Feature Importance

The feature importance plot reveals the key drivers of HDB resale prices. The most influential features are consistently:
1.  `floor_area_sqm`
2.  `remaining_lease_years`
3.  `storey_mid`
4.  Location-based features (e.g., `town_BUKIT MERAH`, `town_QUEENSTOWN`)
5.  `lease_commence_date`

This confirms that the size, age, and location of a flat are the primary determinants of its price.

### Actual vs. Predicted Prices

The scatter plot of actual vs. predicted prices shows a tight clustering of points along the diagonal line, visually confirming the model's high accuracy.

## File Structure

```
HDBprices/
├── ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv
├── eda_hdb_prices.py
├── train_hdb_model_lightbgm.py
└── README.md
```

## How to Run

### Prerequisites

You need Python 3 and the following libraries installed:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn lightgbm xgboost
```

### Running the Scripts

1.  **Run the Exploratory Data Analysis (optional):**
    ```bash
    python eda_hdb_prices.py
    ```
2.  **Train and Evaluate the LightGBM Model:**
    ```bash
    python train_hdb_model_lightbgm.py
    ```

## Future Work

- **Hyperparameter Tuning**: Use techniques like GridSearchCV or RandomizedSearchCV to find the optimal parameters for the LightGBM model to potentially improve accuracy.
- **Advanced Feature Engineering**: Create new features, such as the distance from the flat to the nearest MRT station, school, or shopping mall, which could capture more location-specific value.



