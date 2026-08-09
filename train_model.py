"""
Script to train a Random Forest regression model for used car price prediction.
Supports Kaggle / CarDekho v3 dataset structures automatically.

Produces: model.pkl (joblib) and model_meta.json (form options, metrics & market stats).

Usage:
    python train_model.py [--data data/kaggle_car_data.csv]
"""

import argparse
import json
import os
import sys
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import joblib

# Fix Windows console unicode print encoding
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def parse_and_clean_dataset(df):
    """Normalize dataset columns whether from CarDekho v3, Kaggle, or custom formats."""
    cols = [c.lower().strip() for c in df.columns]
    col_map = {orig: orig.lower().strip() for orig in df.columns}
    df = df.rename(columns=col_map)

    # Check if raw CarDekho structure (name, km_driven, fuel, engine, mileage, owner, etc.)
    if "name" in df.columns and "brand" not in df.columns:
        print("Detected raw CarDekho v3 format. Cleaning and parsing fields...")
        # Extract Brand & Model from 'name'
        df["brand"] = df["name"].apply(lambda x: str(x).split()[0] if pd.notnull(x) else "Unknown")
        df["model"] = df["name"].apply(lambda x: str(x).split()[1] if pd.notnull(x) and len(str(x).split()) > 1 else "Other")

    # Map column aliases
    alias_map = {
        "km_driven": "kilometres_driven",
        "kilometres": "kilometres_driven",
        "km": "kilometres_driven",
        "fuel": "fuel_type",
        "engine": "engine_cc",
        "mileage": "mileage_kmpl",
        "owner": "owner_count",
        "selling_price": "selling_price",
        "price": "selling_price"
    }
    df = df.rename(columns=alias_map)

    # Function to extract numeric values from strings (e.g. '1248 CC' -> 1248.0, '23.4 kmpl' -> 23.4)
    def extract_num(val):
        if pd.isnull(val):
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
        return float(match.group()) if match else np.nan

    if "engine_cc" in df.columns:
        df["engine_cc"] = df["engine_cc"].apply(extract_num)

    if "mileage_kmpl" in df.columns:
        df["mileage_kmpl"] = df["mileage_kmpl"].apply(extract_num)

    if "owner_count" in df.columns:
        def parse_owner(val):
            if isinstance(val, (int, float)):
                return int(val)
            s = str(val).lower()
            if "first" in s or "1st" in s: return 1
            if "second" in s or "2nd" in s: return 2
            if "third" in s or "3rd" in s: return 3
            if "fourth" in s or "4th" in s: return 4
            return 1
        df["owner_count"] = df["owner_count"].apply(parse_owner)

    # Standardize final target columns
    final_rename = {
        "brand": "Brand",
        "model": "Model",
        "year": "Year",
        "kilometres_driven": "Kilometres_Driven",
        "fuel_type": "Fuel_Type",
        "transmission": "Transmission",
        "owner_count": "Owner_Count",
        "engine_cc": "Engine_CC",
        "mileage_kmpl": "Mileage_kmpl",
        "selling_price": "Selling_Price"
    }

    df = df.rename(columns=final_rename)
    return df

def find_best_dataset():
    candidates = [
        "data/kaggle_car_data.csv",
        "data/cardekho_v3_0.csv",
        "data/real_car_data.csv",
        "data/real_car_data_1.csv",
        "data/car_data.csv",
        "SAMPLE_DATASET.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def main(data_path=None):
    if not data_path or not os.path.exists(data_path):
        data_path = find_best_dataset()
        if not data_path:
            raise FileNotFoundError("No dataset CSV file found in workspace.")

    print(f"Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path)
    df = parse_and_clean_dataset(raw_df)

    required_cols = ["Selling_Price", "Brand", "Model", "Year"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Drop missing target & essential rows
    df = df.dropna(subset=required_cols)

    # Numeric conversion
    df["Year"] = df["Year"].astype(int)
    df["Kilometres_Driven"] = pd.to_numeric(df["Kilometres_Driven"], errors="coerce")
    df["Engine_CC"] = pd.to_numeric(df["Engine_CC"], errors="coerce")
    df["Mileage_kmpl"] = pd.to_numeric(df["Mileage_kmpl"], errors="coerce")
    df["Owner_Count"] = pd.to_numeric(df["Owner_Count"], errors="coerce").fillna(1)
    df["Fuel_Type"] = df["Fuel_Type"].fillna("Petrol")
    df["Transmission"] = df["Transmission"].fillna("Manual")

    # Impute missing numeric features with medians
    df["Kilometres_Driven"] = df["Kilometres_Driven"].fillna(df["Kilometres_Driven"].median())
    df["Engine_CC"] = df["Engine_CC"].fillna(df["Engine_CC"].median())
    df["Mileage_kmpl"] = df["Mileage_kmpl"].fillna(df["Mileage_kmpl"].median())

    features = ["Brand", "Model", "Year", "Kilometres_Driven", "Fuel_Type", "Transmission", "Owner_Count", "Engine_CC", "Mileage_kmpl"]
    df = df.dropna(subset=features + ["Selling_Price"]).reset_index(drop=True)

    print(f"Dataset cleaned successfully! Using {len(df)} car records across {len(df['Brand'].unique())} brands.")

    X = df[features]
    y = df["Selling_Price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    categorical_features = ["Brand", "Model", "Fuel_Type", "Transmission"]
    numeric_features = ["Year", "Kilometres_Driven", "Owner_Count", "Engine_CC", "Mileage_kmpl"]

    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", numeric_transformer, numeric_features),
        ]
    )

    regressor = RandomForestRegressor(n_estimators=100, max_depth=16, random_state=42, n_jobs=1)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

    print("Training Random Forest pipeline model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    r2 = float(r2_score(y_test, preds))
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(root_mean_squared_error(y_test, preds))
    mape = float(mean_absolute_percentage_error(y_test, preds)) * 100.0

    print("\n--- Model Evaluation Metrics ---")
    print(f"Dataset File: {data_path}")
    print(f"Records Count: {len(df)}")
    print(f"R2 Accuracy Score: {r2:.4f} ({r2*100:.2f}%)")
    print(f"Mean Absolute Error (MAE): Rs. {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): Rs. {rmse:,.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    # Save trained pipeline
    model_path = "model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved trained model to {model_path}")

    # Feature Importance extraction
    onehot_cols = pipeline.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_features)
    all_feature_names = list(onehot_cols) + numeric_features
    importances = regressor.feature_importances_

    feature_imp_map = {feat: 0.0 for feat in features}
    for name, imp in zip(all_feature_names, importances):
        found = False
        for orig in categorical_features:
            if name.startswith(orig + "_"):
                feature_imp_map[orig] += float(imp)
                found = True
                break
        if not found and name in numeric_features:
            feature_imp_map[name] += float(imp)

    meta = {}
    meta["metrics"] = {
        "r2": round(r2, 4),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "sample_count": len(df),
        "dataset_name": os.path.basename(data_path)
    }

    meta["brands"] = sorted(df["Brand"].unique().tolist())
    meta["models_by_brand"] = {}
    for b in meta["brands"]:
        meta["models_by_brand"][b] = sorted(df[df["Brand"] == b]["Model"].unique().tolist())

    meta["fuel_types"] = sorted(df["Fuel_Type"].dropna().unique().tolist())
    meta["transmissions"] = sorted(df["Transmission"].dropna().unique().tolist())
    meta["owner_counts"] = sorted(df["Owner_Count"].dropna().unique().astype(int).tolist())

    # Price by year
    price_by_year = df.groupby("Year")["Selling_Price"].mean().reset_index().sort_values("Year")
    meta["price_by_year"] = price_by_year.to_dict(orient="records")

    # Price by brand (top 15)
    price_by_brand = df.groupby("Brand")["Selling_Price"].mean().reset_index().sort_values("Selling_Price", ascending=False).head(15)
    meta["price_by_brand"] = price_by_brand.to_dict(orient="records")

    meta["feature_importances"] = [
        {"feature": feat, "importance": round(imp * 100, 2)}
        for feat, imp in sorted(feature_imp_map.items(), key=lambda x: x[1], reverse=True)
    ]

    meta["feature_ranges"] = {
        "year": {"min": int(df["Year"].min()), "max": int(df["Year"].max())},
        "kilometres": {"min": int(df["Kilometres_Driven"].min()), "max": int(df["Kilometres_Driven"].max())},
        "engine_cc": {"min": float(df["Engine_CC"].min()), "max": float(df["Engine_CC"].max())},
        "mileage": {"min": float(df["Mileage_kmpl"].min()), "max": float(df["Mileage_kmpl"].max())}
    }

    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved updated metadata to model_meta.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", default=None, help="Path to CSV dataset")
    args = parser.parse_args()
    main(args.data)
