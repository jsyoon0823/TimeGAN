
```python
import pandas as pd
import numpy as np

cities = ['Baabda', 'Beirut', 'Bekaa', 'Kesrouan', 'Tripoli']
all_cities_df = []

print("Processing and combining synthetic data for all cities...")

for city in cities:
    print(f"...Processing {city}")
    
    # 1. Load the specific files for this city
    df_synthetic_scaled = pd.read_csv(f'synthetic_data_{city}.csv')
    df_synthetic_scaled.columns = ['transaction_number', 'transaction_value']
    
    min_max_data = np.load(f'min_max_{city}.npz')
    min_val = min_max_data['min_val']
    max_val = min_max_data['max_val']
    
    # 2. Un-scale the data
    df_synthetic_denormalized = df_synthetic_scaled * (max_val - min_val) + min_val
    
    # 3. Create the date range and add the city column
    date_range = pd.date_range(start='2017-01-01', periods=len(df_synthetic_denormalized), freq='MS')
    df_synthetic_denormalized['date'] = date_range
    df_synthetic_denormalized['city'] = city
    
    # 4. Append to our master list
    all_cities_df.append(df_synthetic_denormalized)

# 5. Concatenate all city dataframes into one final dataframe
final_df = pd.concat(all_cities_df, ignore_index=True)

# Reorder columns for clarity
final_df = final_df[['date', 'city', 'transaction_number', 'transaction_value']]

# 6. Save the final, correct file
output_file = 'final_usable_synthetic_data_COMBINED.csv'
final_df.to_csv(output_file, index=False)

print(f"\nSuccess! All cities combined into '{output_file}'")
print("\nHere is a preview:")
print(final_df.head())