Project Report: Used Car Price Prediction
=========================================

1. Introduction
---------------
This project implements a web application that predicts the selling price of used cars using a Random Forest regression model. The application includes a responsive web UI, REST API, SQLite-backed history logging, and charts for trends and depreciation.

2. Data
-------
The model expects a CSV with the following columns:
- Brand, Model, Year, Kilometres_Driven, Fuel_Type, Transmission, Owner_Count, Engine_CC, Mileage_kmpl, Selling_Price

SAMPLE_DATASET.csv in the repository shows an example format.

3. Methodology
--------------
- Preprocessing: One-hot encoding for categorical features (Brand, Model, Fuel_Type, Transmission) and standard scaling for numeric features.
- Model: RandomForestRegressor with 100 trees.
- Evaluation: Train/test split with R2 and MAE printed after training.

4. Deployment
-------------
The trained pipeline is saved using joblib as model.pkl, and the metadata used to populate UI inputs is saved in model_meta.json. The Flask app loads the model at startup and exposes both a web UI and REST API.

5. UI & UX
---------
- Bootstrap 4 used for responsive layout.
- Client-side JS dynamically populates model options based on brand selection.
- Results include estimated price, range, confidence, and a recommendation.

6. Limitations & Future Work
----------------------------
- The model quality depends strongly on dataset size and feature quality.
- Future improvements: better feature engineering, hyperparameter tuning, cross-validation, and deployment using a WSGI server (gunicorn) behind nginx.

7. How to Reproduce
-------------------
See README.md for setup, training, and running instructions.

8. Conclusion
-------------
A starter, production-ready structure for a used car price prediction app has been provided with clear separation of concerns and simple extensibility paths.
