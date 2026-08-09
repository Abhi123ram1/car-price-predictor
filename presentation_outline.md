Presentation Outline: Used Car Price Prediction
===============================================

Slide 1: Title
- Project name, author (GitHub Copilot), date

Slide 2: Problem Statement
- Estimating fair selling price for used cars
- Use cases: sellers, buyers, dealerships

Slide 3: Data & Features
- Mention key features used: Brand, Model, Year, Kilometres, Fuel, Transmission, Owner Count, Engine, Mileage

Slide 4: Model & Approach
- Random Forest Regressor
- Preprocessing: One-hot encoding, scaling
- Train/test split, saved pipeline

Slide 5: System Architecture
- train_model.py -> model.pkl + model_meta.json
- Flask app loads model, serves UI & API
- SQLite stores prediction history

Slide 6: Demo Screens
- Home, Predict form, Dashboard charts

Slide 7: Metrics & Evaluation
- Show R2, MAE (from training)

Slide 8: Recommendations & Business Logic
- Confidence score, price range, recommendation rules

Slide 9: Future Enhancements
- Hyperparameter tuning, more data, deployment tips

Slide 10: Q&A
- Contact info
