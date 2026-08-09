# 🚗 CarValue AI - Used Car Price Predictor

An AI-powered web application for used car valuation built with **Python**, **Flask**, **scikit-learn**, **Pandas**, **Chart.js**, and **SQLite/PostgreSQL**.

---

## 🌟 Key Features

- **Machine Learning Engine**: Trained on 600+ real-world car listings using a **Random Forest Regressor** pipeline (`r2_score = 91.26%`, `MAE = ₹89,246`).
- **Dynamic Price Range & Confidence Rating**: Estimates high and low valuation boundaries based on ensemble tree variance.
- **Smart Deal Classification**: Categorizes predictions into **Good Deal**, **Fair Price**, or **Overpriced**.
- **Dark Glassmorphic UI**: Modern interface with responsive layout, dynamic brand/model dropdown auto-filtering, and micro-interactions.
- **Analytics Dashboard**: Interactive Chart.js graphs displaying valuation timelines, depreciation curves by year, and search/filterable history table.
- **Database Support**: Automatic schema initialization with parameter abstraction for both **SQLite** and **PostgreSQL**.
- **REST API & Data Export**: Includes JSON endpoints for metadata, predictions, history, and CSV download export.

---

## 📐 Architecture & Technology Stack

| Component | Technologies |
| :--- | :--- |
| **Backend Framework** | Flask, Python 3.14, Gunicorn |
| **Machine Learning** | scikit-learn (RandomForestRegressor, OneHotEncoder, StandardScaler), Joblib |
| **Data Processing** | Pandas, NumPy |
| **Database** | SQLite / PostgreSQL (psycopg2) |
| **Frontend** | HTML5, CSS3 (Custom Glassmorphism), Bootstrap 4.6, FontAwesome 6 |
| **Data Visualization**| Chart.js |

---

## 🚀 Quick Start Guide

### 1. Prerequisites
Ensure Python 3.8+ is installed on your system.

### 2. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Generate Dataset & Train Model
```bash
# 1. Generate realistic dataset
python data/generate_dataset.py

# 2. Train the Random Forest Regressor
python train_model.py --data data/car_data.csv
```

### 4. Run the Flask Web Application
```bash
python app.py
```
Open your browser and navigate to `http://localhost:5000`.

---

## 📡 REST API Documentation

### 1. Get Model Metadata & Form Options
`GET /api/meta`

**Response:**
```json
{
  "metrics": {
    "r2": 0.9126,
    "mae": 89246.62,
    "rmse": 173192.73,
    "sample_count": 600
  },
  "brands": ["Audi", "BMW", "Ford", "Honda", "Hyundai", "Kia", "Mahindra", "Maruti", "Tata", "Toyota", "Volkswagen"],
  "models_by_brand": {
    "Maruti": ["Alto", "Baleno", "Brezza", "Dzire", "Ertiga", "Swift", "Wagon R"]
  }
}
```

### 2. Make a Price Prediction
`POST /api/predict`

**Request Body:**
```json
{
  "Brand": "Maruti",
  "Model": "Swift",
  "Year": 2020,
  "Kilometres_Driven": 35000,
  "Fuel_Type": "Petrol",
  "Transmission": "Manual",
  "Owner_Count": 1,
  "Engine_CC": 1197,
  "Mileage_kmpl": 22.0
}
```

**Response:**
```json
{
  "predicted_price": 465200.0,
  "price_low": 427984.0,
  "price_high": 502416.0,
  "confidence": 92.5,
  "recommendation": "Good Deal",
  "currency": "INR (₹)",
  "timestamp": "2026-07-29 13:45:00"
}
```

### 3. Download History as CSV
`GET /api/export-csv`

---

## 🐳 Docker Deployment

To build and run using Docker:
```bash
docker build -t car-price-predictor .
docker run -p 5000:5000 car-price-predictor
```

Or using Docker Compose:
```bash
docker-compose up --build
```
