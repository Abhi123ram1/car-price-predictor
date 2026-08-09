"""
Synthetic Dataset Generator for Used Car Price Prediction.
Generates realistic Indian car market data with 500+ records.
"""

import pandas as pd
import numpy as np
import os

def generate_car_dataset(n_samples=600, random_seed=42):
    np.random.seed(random_seed)

    brands_models = {
        "Maruti": [
            ("Swift", 1197, 22.0, 450000, "Petrol"),
            ("Baleno", 1197, 22.35, 550000, "Petrol"),
            ("Dzire", 1197, 23.26, 600000, "Petrol"),
            ("Ertiga", 1462, 19.0, 850000, "Petrol"),
            ("Brezza", 1462, 18.76, 900000, "Petrol"),
            ("Alto", 796, 22.05, 300000, "Petrol"),
            ("Wagon R", 998, 21.79, 400000, "CNG")
        ],
        "Hyundai": [
            ("i20", 1197, 20.35, 650000, "Petrol"),
            ("Verna", 1497, 18.6, 950000, "Petrol"),
            ("Creta", 1497, 17.0, 1100000, "Petrol"),
            ("Venue", 998, 18.2, 800000, "Petrol"),
            ("Grand i10", 1197, 20.7, 500000, "Petrol")
        ],
        "Honda": [
            ("City", 1498, 17.8, 1000000, "Petrol"),
            ("Amaze", 1199, 18.6, 700000, "Petrol"),
            ("WR-V", 1498, 17.5, 750000, "Petrol"),
            ("Civic", 1799, 16.5, 1400000, "Petrol")
        ],
        "Toyota": [
            ("Innova Crysta", 2393, 15.1, 1600000, "Diesel"),
            ("Fortuner", 2755, 10.0, 2800000, "Diesel"),
            ("Glanza", 1197, 22.3, 600000, "Petrol"),
            ("Urban Cruiser", 1462, 18.7, 850000, "Petrol")
        ],
        "Tata": [
            ("Nexon", 1199, 17.0, 850000, "Petrol"),
            ("Harrier", 1956, 16.3, 1500000, "Diesel"),
            ("Safari", 1956, 16.1, 1700000, "Diesel"),
            ("Tiago", 1199, 20.0, 500000, "Petrol"),
            ("Punch", 1199, 18.9, 650000, "Petrol")
        ],
        "Mahindra": [
            ("Thar", 2184, 15.2, 1300000, "Diesel"),
            ("XUV700", 2184, 15.0, 1600000, "Diesel"),
            ("Scorpio-N", 2184, 14.5, 1500000, "Diesel"),
            ("XUV300", 1197, 17.0, 850000, "Petrol"),
            ("Bolero", 1493, 16.0, 750000, "Diesel")
        ],
        "Ford": [
            ("EcoSport", 1496, 15.9, 750000, "Petrol"),
            ("Endeavour", 1996, 12.4, 2500000, "Diesel"),
            ("Figo", 1194, 18.5, 450000, "Petrol")
        ],
        "BMW": [
            ("3 Series", 1998, 16.1, 3500000, "Petrol"),
            ("5 Series", 1998, 15.5, 4500000, "Petrol"),
            ("X1", 1995, 16.3, 3000000, "Diesel"),
            ("X5", 2993, 11.2, 6000000, "Diesel")
        ],
        "Audi": [
            ("A4", 1984, 17.4, 3200000, "Petrol"),
            ("A6", 1984, 14.1, 4200000, "Petrol"),
            ("Q3", 1984, 15.8, 2800000, "Petrol"),
            ("Q5", 1984, 13.4, 4500000, "Petrol")
        ],
        "Kia": [
            ("Seltos", 1497, 16.5, 1100000, "Petrol"),
            ("Sonet", 1197, 18.4, 800000, "Petrol"),
            ("Carens", 1497, 16.5, 1000000, "Petrol")
        ],
        "Volkswagen": [
            ("Polo", 999, 18.2, 550000, "Petrol"),
            ("Vento", 999, 17.6, 750000, "Petrol"),
            ("Taigun", 999, 18.1, 1050000, "Petrol")
        ]
    }

    current_year = 2026
    data = []

    for _ in range(n_samples):
        brand = np.random.choice(list(brands_models.keys()))
        model_info = brands_models[brand][np.random.randint(0, len(brands_models[brand]))]
        model_name, base_cc, base_mileage, base_price, default_fuel = model_info

        # Year between 2011 and 2025
        year = int(np.random.randint(2011, 2026))
        age = current_year - year

        # Kilometres driven increases with age
        kms = int(np.random.normal(loc=12000 * age + 5000, scale=8000))
        kms = max(2000, min(250000, kms))

        # Fuel type
        fuel_choices = ["Petrol", "Diesel", "CNG", "Electric"]
        fuel_weights = [0.55, 0.35, 0.07, 0.03]
        fuel_type = np.random.choice(fuel_choices, p=fuel_weights)

        # Transmission
        transmission = np.random.choice(["Manual", "Automatic"], p=[0.65, 0.35])

        # Owner count (1 to 3, older cars tend to have more owners)
        if age <= 3:
            owner_count = np.random.choice([1, 2], p=[0.9, 0.1])
        elif age <= 7:
            owner_count = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        else:
            owner_count = np.random.choice([1, 2, 3, 4], p=[0.3, 0.4, 0.2, 0.1])

        # Engine CC slight variance
        engine_cc = float(round(base_cc * np.random.uniform(0.95, 1.05), 1))

        # Mileage kmpl variance
        mileage = float(round(base_mileage * np.random.uniform(0.9, 1.1), 1))

        # Price calculation with realistic depreciation formula
        # Base price * (0.88 ^ age) * kms penalty * owner penalty * transmission bonus
        depreciation_factor = (0.86 ** age)
        kms_penalty = max(0.55, 1.0 - (kms / 350000.0))
        owner_penalty = 1.0 - (owner_count - 1) * 0.08
        trans_bonus = 1.12 if transmission == "Automatic" else 1.0
        fuel_bonus = 1.05 if fuel_type == "Diesel" else (0.95 if fuel_type == "CNG" else 1.0)

        price = base_price * depreciation_factor * kms_penalty * owner_penalty * trans_bonus * fuel_bonus
        # Add random market noise (+/- 7%)
        price = price * np.random.uniform(0.93, 1.07)
        price = float(round(max(40000, price), -3)) # round to thousands

        data.append({
            "Brand": brand,
            "Model": model_name,
            "Year": year,
            "Kilometres_Driven": kms,
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
            "Owner_Count": owner_count,
            "Engine_CC": engine_cc,
            "Mileage_kmpl": mileage,
            "Selling_Price": price
        })

    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    csv_path = os.path.join("data", "car_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} car listing records saved to {csv_path}")
    return csv_path

if __name__ == "__main__":
    generate_car_dataset()
