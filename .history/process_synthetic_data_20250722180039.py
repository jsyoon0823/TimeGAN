import pandas as pd
import numpy as np

# --- IMPORTANT: YOU MUST CUSTOMIZE THIS SECTION ---

# TODO: Define the column names in the exact order they went into the model.
# To find this order, you can add `print(df.columns)` to your `transaction_data_loading`
# function right before the `ori_data = df.values` line and run the main script again.
# The console output will give you the exact list you need here.
#
# EXAMPLE:
# column_names = ['transaction_value', 'feature_2', 'city_1', 'city_2', 'city_3', 'city_4', 'city_5']
#
column_names = [
    'transaction_number',
    'transaction_value',
    'city_Baabda',      # Alphabetical order is crucial
    'city_Bekaa',
    'city_Beirut',
    'city_Kesrouan',
    'city_Tripoli'
]


original_city_column_name = 'city'

# TODO: Define the name of your original date and id columns from the CSV.
original_date_column_name = 'date'
original_id_column_name = 'id'



print("Starting the reverse transformation process...")

# --- Step 1: Load and Rename the Raw Synthetic Data ---
try:
    # THIS IS THE FIX: Only read the first 360 rows.
    df_synthetic = pd.read_csv('synthetic_data.csv', nrows=360)
    
    # The raw CSV has headers '0', '1', '2', etc. We replace them.
    df_synthetic.columns = column_names
    print("-> Successfully loaded and renamed raw synthetic data.")
except FileNotFoundError:
    print("Error: 'synthetic_data.csv' not found. Please run main_timegan.py first to generate the data.")
    exit()


# --- Step 2: Calculate Min/Max Values from Original Data for Denormalization ---
df_original = pd.read_csv('data/transaction_data.csv')
if original_date_column_name in df_original.columns:
    df_original = df_original.drop(columns=[original_date_column_name])
if original_id_column_name in df_original.columns:
    df_original = df_original.drop(columns=[original_id_column_name])
if original_city_column_name in df_original.columns:
    df_original = pd.get_dummies(df_original, columns=[original_city_column_name], prefix='city')

df_original = df_original[column_names]
min_vals = df_original.min(axis=0)
max_vals = df_original.max(axis=0)
print("-> Calculated min/max values from original data for denormalization.")


# --- Step 3: Denormalize the Synthetic Data ---
df_synthetic_denormalized = df_synthetic.copy()
for col in column_names:
    df_synthetic_denormalized[col] = df_synthetic[col] * (max_vals[col] - min_vals[col]) + min_vals[col]
print("-> Synthetic data has been denormalized to its original scale.")


# --- Step 4: Reconstruct the 'city' Column from One-Hot Encoding ---
city_cols = [col for col in column_names if col.startswith('city_')]
df_synthetic_denormalized['city'] = df_synthetic_denormalized[city_cols].idxmax(axis=1).str.replace('city_', '')
df_final = df_synthetic_denormalized.drop(columns=city_cols)
print("-> Reconstructed 'city' column with text names.")


# --- Step 5: Add a Date Index ---
date_range = pd.date_range(start='2017-01-01', periods=72, freq='M')
df_final.set_index(date_range, inplace=True)
df_final.index.name = 'date'
print("-> Added a date index.")


# --- Step 6: Save the Final, Usable Data ---
df_final.to_csv('final_usable_synthetic_data.csv')
print("\nSuccess! Your final, usable synthetic data has been saved to 'final_usable_synthetic_data.csv'")
print("\nHere is a preview:")
print(df_final.head())
