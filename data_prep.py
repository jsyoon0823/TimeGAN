import pandas as pd

# Load your cleaned, original data file
df = pd.read_csv('data/transaction_data.csv') 

# Get a list of unique cities
cities = df['city'].unique()

# Create a separate CSV file for each city
for city in cities:
    city_df = df[df['city'] == city]
    # We only need the numeric columns for training
    city_df[['transaction_number', 'transaction_value']].to_csv(f'data/{city}_data.csv', index=False)
    print(f'Created data/{city}_data.csv')