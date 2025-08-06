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
# Replace the example below with your actual column names:
column_names = [
    'transaction_number',
    'transaction_value',
    'city_Baabda',      # Alphabetical order
    'city_Baalbak',     #
    'city_Bekaa',       # This is newly added
    'city_Beirut',      #
    'city_Kesrouan',    #
    'city_Tripoli'      #
]


original_city_column_name = 'city'

# TODO: Define the name of your original date and id columns from the CSV.
original_date_column_name = 'date'
original_id_column_name = 'id'

# --- THE REST OF THE SCRIPT SHOULD WORK AUTOMATICALLY ---

print("Starting the reverse transformation process...")

# --- Step 1: Load and Rename the Raw Synthetic Data ---
try:
    df_synthetic = pd.read_csv('synthetic_data.csv')
    # The raw CSV has headers '0', '1', '2', etc. We replace them.
    df_synthetic.columns = column_names
    print("-> Successfully loaded and renamed raw synthetic data.")
except FileNotFoundError:
    print("Error: 'synthetic_data.csv' not found. Please run main_timegan.py first to generate the data.")
    exit()


# --- Step 2: Calculate Min/Max Values from Original Data for Denormalization ---

# Load and process the original data exactly as the GAN did
df_original = pd.read_csv('data/transaction_data.csv')
if original_date_column_name in df_original.columns:
    df_original = df_original.drop(columns=[original_date_column_name])
if original_city_column_name in df_original.columns:
    df_original = pd.get_dummies(df_original, columns=[original_city_column_name], prefix='city')

# Ensure the column order matches what the model was trained on
df_original = df_original[column_names]

# Calculate the min and max values for each column
min_vals = df_original.min(axis=0)
max_vals = df_original.max(axis=0)
print("-> Calculated min/max values from original data for denormalization.")


# --- Step 3: Denormalize the Synthetic Data ---
# Formula: original_value = normalized_value * (max - min) + min
df_synthetic_denormalized = df_synthetic.copy()
for col in column_names:
    df_synthetic_denormalized[col] = df_synthetic[col] * (max_vals[col] - min_vals[col]) + min_vals[col]
print("-> Synthetic data has been denormalized to its original scale.")


# --- Step 4: Reconstruct the 'city_id' Column from One-Hot Encoding ---
city_cols = [col for col in column_names if col.startswith('city_')]

# For each row, find which 'city_X' column has the highest value
# Then, extract the number 'X' to be the city_id
df_synthetic_denormalized['id'] = df_synthetic_denormalized[city_cols].idxmax(axis=1).str.replace('city_', '').astype(int)

# Drop the now-redundant one-hot encoded columns
df_final = df_synthetic_denormalized.drop(columns=city_cols)
print("-> Reconstructed 'city_id' column.")


# --- Step 5: Add a Date Index ---
# The GAN generates a sequence. We need to give it a realistic time index.
# We create a date range that matches the length of the generated data.
# Customize the start date and frequency ('M' for month, 'D' for day) as needed.
date_range = pd.date_range(start='2017-01-01', periods=len(df_final), freq='M')
df_final.set_index(date_range, inplace=True)
df_final.index.name = 'date'
print("-> Added a date index.")


# --- Step 6: Save the Final, Usable Data ---
df_final.to_csv('final_usable_synthetic_data.csv')
print("\nSuccess! Your final, usable synthetic data has been saved to 'final_usable_synthetic_data.csv'")
print("\nHere is a preview:")
print(df_final.head())