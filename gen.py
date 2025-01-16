import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)


# Define sample data
makes = ['Toyota', 'Honda', 'Ford', 'BMW', 'Mercedes', 'Volkswagen', 'Audi', 'Hyundai']
models = {
    'Toyota': ['Camry', 'Corolla', 'RAV4', 'Highlander', 'Prius'],
    'Honda': ['Civic', 'Accord', 'CR-V', 'Pilot', 'HR-V'],
    'Ford': ['F-150', 'Explorer', 'Escape', 'Mustang', 'Edge'],
    'BMW': ['3 Series', '5 Series', 'X3', 'X5', '7 Series'],
    'Mercedes': ['C-Class', 'E-Class', 'GLC', 'GLE', 'S-Class'],
    'Volkswagen': ['Golf', 'Passat', 'Tiguan', 'Atlas', 'Jetta'],
    'Audi': ['A3', 'A4', 'Q5', 'Q7', 'A6'],
    'Hyundai': ['Tucson', 'Santa Fe', 'Elantra', 'Kona', 'Palisade']
}


# Generate random dates within a range
def random_date(start_date, end_date):
    time_between = end_date - start_date
    days_between = time_between.days
    random_days = np.random.randint(0, days_between)
    return start_date + timedelta(days=random_days)


# Generate sample data
n_samples = 100
data = []

start_date = datetime(2020, 1, 1)
end_date = datetime(2024, 1, 1)

for i in range(n_samples):
    make = np.random.choice(makes)
    model = np.random.choice(models[make])
    fab_year = np.random.randint(2018, 2024)

    # Generate VIN-like identifier (simplified)
    vehicle_id = f"{make[:3].upper()}{fab_year}{i:03d}"

    data.append({
        'vehicle': vehicle_id,
        'date_bought': random_date(start_date, end_date).strftime('%Y-%m-%d'),
        'make': make,
        'fabrication_year': f"{fab_year}-01-01",
        'model': model
    })

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('sample_vehicles.csv', index=False)

# Display first few rows
print("First few rows of the generated data:")
print(df.head())

# Display summary statistics
print("\nSummary of the generated data:")
print(f"Total number of vehicles: {len(df)}")
print("\nVehicles by make:")
print(df['make'].value_counts())
print("\nDate range:")
print(f"Earliest purchase date: {df['date_bought'].min()}")
print(f"Latest purchase date: {df['date_bought'].max()}")
