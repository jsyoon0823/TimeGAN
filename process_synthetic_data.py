# import pandas as pd
# import numpy as np

# # --- IMPORTANT: YOU MUST CUSTOMIZE THIS SECTION ---

# # TODO: Define the column names in the exact order they went into the model.
# # To find this order, you can add `print(df.columns)` to your `transaction_data_loading`
# # function right before the `ori_data = df.values` line and run the main script again.
# # The console output will give you the exact list you need here.
# #
# # EXAMPLE:
# # column_names = ['transaction_value', 'feature_2', 'city_1', 'city_2', 'city_3', 'city_4', 'city_5']
# #
# column_names = [
#     'transaction_number',
#     'transaction_value',
#     'city_Baabda',      # Alphabetical order is crucial
#     'city_Bekaa',
#     'city_Beirut',
#     'city_Kesrouan',
#     'city_Tripoli'
# ]


# original_city_column_name = 'city'

# # TODO: Define the name of your original date and id columns from the CSV.
# original_date_column_name = 'date'
# original_id_column_name = 'id'



# print("Starting the reverse transformation process...")

# # --- Step 1: Load Raw Synthetic Data ---
# try:
#     df_synthetic = pd.read_csv('synthetic_data.csv', nrows=360)
#     df_synthetic.columns = column_names
#     print("-> Successfully loaded and renamed raw synthetic data.")
# except FileNotFoundError:
#     print("Error: 'synthetic_data.csv' not found. Please run main_timegan.py first to generate the data.")
#     exit()


# # --- Step 2: Calculate Min/Max for Denormalization ---
# df_original = pd.read_csv('data/transaction_data.csv')
# if original_date_column_name in df_original.columns:
#     df_original = df_original.drop(columns=[original_date_column_name])
# if original_id_column_name in df_original.columns:
#     df_original = df_original.drop(columns=[original_id_column_name])
# if original_city_column_name in df_original.columns:
#     df_original = pd.get_dummies(df_original, columns=[original_city_column_name], prefix='city')

# df_original = df_original[column_names]
# min_vals = df_original.min(axis=0)
# max_vals = df_original.max(axis=0)
# print("-> Calculated min/max values from original data.")


# # --- Step 3: Denormalize Synthetic Data ---
# df_synthetic_denormalized = df_synthetic.copy()
# for col in column_names:
#     df_synthetic_denormalized[col] = df_synthetic[col] * (max_vals[col] - min_vals[col]) + min_vals[col]
# print("-> Synthetic data has been denormalized.")


# # --- Step 4: Reconstruct 'city' Column ---
# city_cols = [col for col in column_names if col.startswith('city_')]
# df_synthetic_denormalized['city'] = df_synthetic_denormalized[city_cols].idxmax(axis=1).str.replace('city_', '')
# df_final_data_only = df_synthetic_denormalized.drop(columns=city_cols)
# print("-> Reconstructed 'city' column with text names.")


# # --- Step 5: Create a Structured DataFrame with Correct Dates ---
# # THIS IS THE FIX: Use 'MS' for Month Start frequency to match your original data.
# date_range_single_city = pd.date_range(start='2017-01-01', periods=72, freq='MS')

# cities = sorted(df_final_data_only['city'].unique()) # Using the correct df

# structured_dfs = []
# for city_name in cities:
#     temp_df = pd.DataFrame({
#         'date': date_range_single_city,
#         'city': city_name
#     })
#     structured_dfs.append(temp_df)

# final_index_df = pd.concat(structured_dfs, ignore_index=True)

# df_final_data_only.reset_index(drop=True, inplace=True)
# final_index_df.reset_index(drop=True, inplace=True)

# # Drop the flawed city column from the data part before combining
# df_final_data_only = df_final_data_only.drop(columns=['city'])
# df_final_structured = pd.concat([final_index_df, df_final_data_only], axis=1)

# print("-> Created a structured final DataFrame.")


# # --- Step 6: Save the Final Data ---
# df_final_structured.to_csv('final_usable_synthetic_data.csv', index=False)
# print("\nSuccess! Your final, usable synthetic data has been saved to 'final_usable_synthetic_data.csv'")
# print("\nHere is a preview:")
# print(df_final_structured.head())

import pandas as pd
import numpy as np

# --- Customize this section with your column names ---
column_names = [
    'transaction_number',
    'transaction_value',
    'city_Baabda',
    'city_Bekaa',
    'city_Beirut',
    'city_Kesrouan',
    'city_Tripoli'
]
# ------------------------------------

print("Starting the un-scaling process...")

# --- Step 1: Load the raw synthetic data and our saved min/max values ---
try:
    df_synthetic_scaled = pd.read_csv('synthetic_data.csv')
    df_synthetic_scaled.columns = column_names
    
    # Load the min and max values we saved from training
    min_max_data = np.load('min_max_values.npz')
    min_val = min_max_data['min_val']
    max_val = min_max_data['max_val']
    
    print("-> Loaded synthetic data and min/max values successfully.")
    
except FileNotFoundError as e:
    print(f"Error: {e}. Did you run main_timegan.py after modifying data_loading.py?")
    exit()

# --- Step 2: Un-scale the data using the correct formula ---
# This is the inverse of the formula: (data - min) / (max - min)
df_synthetic_denormalized = df_synthetic_scaled * (max_val - min_val) + min_val
print("-> Un-scaling complete.")

# --- Step 3: Reconstruct the 'city' column ---
city_cols = [col for col in column_names if col.startswith('city_')]
df_synthetic_denormalized['city'] = df_synthetic_denormalized[city_cols].idxmax(axis=1).str.replace('city_', '')
df_final_data_only = df_synthetic_denormalized.drop(columns=city_cols)
print("-> Reconstructed the 'city' column.")

# --- Step 4: Create the final, structured DataFrame with dates ---
# Get the number of cities and the number of time steps per city
num_cities = len(df_final_data_only['city'].unique())
timesteps_per_city = len(df_final_data_only) // num_cities

date_range = pd.date_range(start='2017-01-01', periods=timesteps_per_city, freq='MS')
cities = sorted(df_final_data_only['city'].unique())
full_date_range = np.tile(date_range, num_cities)
full_city_list = np.repeat(cities, len(date_range))

df_final_structured = pd.DataFrame({'date': full_date_range, 'city': full_city_list})

# Sort the data by city to ensure it aligns correctly before concatenating
df_final_data_only = df_final_data_only.sort_values(by=['city']).reset_index(drop=True)
df_final_structured = pd.concat([df_final_structured, df_final_data_only.drop('city', axis=1)], axis=1)
print("-> Created the final structured DataFrame.")

# --- Step 5: Save the final, usable data ---
output_file = 'final_usable_synthetic_data.csv'
df_final_structured.to_csv(output_file, index=False)
print(f"\nSuccess! Your final data is saved in '{output_file}'")
print("\nHere is a preview:")
print(df_final_structured.head())