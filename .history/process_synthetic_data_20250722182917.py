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

