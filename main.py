
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# LOAD THE DATASET

df = pd.read_csv("House Price India.csv" )

print("✅ Dataset loaded successfully")

# 2.EXPLORE DATASET STRUCTURE & PROPERTIES

def show_info(df):
    print("\n[STEP 2] BASIC INFORMATION ABOUT THE DATA")
    print("-" * 60)

    # First 5 rows
    print("\nFirst 5 rows:")
    print(df.head())

    # Shape of dataset
    print("\nShape (rows, columns):")
    print(df.shape)

    # Column names
    print("\nColumn names:")
    print(df.columns.tolist())

    # Column data types
    print("\nColumn data types:")
    print(df.dtypes)

    # Missing values per column
    print("\nMissing values per column:")
    print(df.isna().sum())

    # Unique value count
    print("\nUnique values per column:")
    print(df.nunique())

    # Duplicate rows
    print("\nNumber of duplicate rows:")
    print(df.duplicated().sum())

    # Summary statistics for numeric columns
    print("\nSummary statistics (numeric columns):")
    print(df.describe())

show_info(df)

# STEP 3: Identify Numerical & Categorical Columns

print("\n[STEP 3] Identifying column types...")
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_like = [col for col in numerical_cols if df[col].nunique() <= 15]

print("\nCategorical-like Numeric Columns:")
for col in categorical_like:
    print("  -", col)

# 4. HANDLE MISSING VALUES

def handle_missing_values(df):
    print("\n[STEP 4] HANDLING MISSING VALUES")
    print("-" * 60)

    # Missing values before cleaning
    print("\nMissing values BEFORE cleaning:")
    print(df.isna().sum())

    df_clean = df.copy()

    # If no missing values
    if df_clean.isna().sum().sum() == 0:
        print("\nNo missing values found.")
        return df_clean

    print("\nMissing values detected. Applying hybrid cleaning...")

    # Identify column types
    numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_clean.select_dtypes(include=['object', 'category', 'bool']).columns

    # Fill numeric columns with median
    for col in numerical_cols:
        median_value = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_value)

    # Fill categorical columns with mode
    for col in categorical_cols:
        mode_value = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_value)

    # Missing values after cleaning
    print("\nMissing values AFTER cleaning:")
    print(df_clean.isna().sum())

    return df_clean

df = handle_missing_values(df)

# STEP 5: Generate Descriptive Statistics (Hybrid Approach)

def show_statistics(df):
    print("\n[STEP 5] DESCRIPTIVE STATISTICS")
    print("-" * 60)

    # Identify numeric and categorical columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Identify categorical-like numeric columns (few unique values)
    categorical_like = [col for col in numerical_cols if df[col].nunique() <= 15]
    continuous_numeric = [col for col in numerical_cols if col not in categorical_like]

    # Continuous numeric summary
    if continuous_numeric:
        print("\n🔹 Continuous Numeric Features:")
        print(df[continuous_numeric].describe().T)
    else:
        print("\nNo continuous numeric columns identified.")

    # Categorical-like numeric summary
    if categorical_like:
        print("\n🔹 Frequency of Categorical-like Numeric Features:")
        for col in categorical_like:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts())
    else:
        print("\nNo categorical-like numeric columns identified.")

    # True categorical summary
    if categorical_cols:
        print("\n🔹 Frequency of Categorical Features:")
        for col in categorical_cols:
            print(f"\nValue counts for {col}:")
            print(df[col].value_counts())
    else:
        print("\nNo categorical columns identified.")

    return df

df = show_statistics(df)

# ==========================================================
# STEP 6: Visualize Key Features
# ==========================================================

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_data(df):
    print("\n[STEP 6] VISUALIZATION")
    print("-" * 60)

    # Histogram of Price
    if "Price" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["Price"], bins=40, kde=True)
        plt.title("Distribution of House Prices")
        plt.xlabel("Price")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    # Scatter plot: Area vs Price (your dataset uses "Area", not "living area")
    if "Area" in df.columns and "Price" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.scatterplot(x="Area", y="Price", data=df, alpha=0.5)
        plt.title("Area vs Price")
        plt.xlabel("Area")
        plt.ylabel("Price")
        plt.tight_layout()
        plt.show()

    # Correlation heatmap with Price
    if "Price" in df.columns:
        corr = df.corr(numeric_only=True)[["Price"]].sort_values(by="Price", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="Blues")
        plt.title("Correlation of Features with Price")
        plt.tight_layout()
        plt.show()

visualize_data(df)

# ==========================================================
# STEP 7: Extract Insights
# ==========================================================

def extract_insights(df):
    print("\n[STEP 7] KEY INSIGHTS")
    print("-" * 60)

    # 1. Dataset size
    print("\n1. Dataset Size:")
    print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    # 2. Correlation with Price
    if "Price" in df.columns:
        print("\n2. Top Features Correlated with Price:")
        price_corr = df.corr(numeric_only=True)["Price"].sort_values(ascending=False)
        print(price_corr.head(10))

    # 3. Outlier detection (simple rule-of-thumb)
    print("\n3. Outlier Check:")
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if df[col].max() > df[col].mean() * 3:
            print(f"   - {col}: possible outliers detected")

    # 4. Missing values summary
    print("\n4. Missing Value Summary:")
    print(df.isna().sum())

extract_insights(df)
