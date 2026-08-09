# 🚗 CarValue AI - Used Car Price Predictor

[![Live App on Render](https://img.shields.io/badge/Live_App-Render-success?style=for-the-badge&logo=render)](https://car-price-predictor-r4a7.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)

**🌐 Live Web Application**: [https://car-price-predictor-r4a7.onrender.com](https://car-price-predictor-r4a7.onrender.com)

An AI-powered web application for used car valuation built with **Python**, **Flask**, **scikit-learn**, **Pandas**, **Chart.js**, and **SQLite/PostgreSQL**.

---

## 🌟 Key Features

- **High-Accuracy ML Model**: Trained on 8,128+ real-world car listings across 32 brands using a **Random Forest Regressor** pipeline (`R² Score = 96.52%`, `MAE = ₹70,429`).
- **Dynamic Price Range & Confidence Rating**: Estimates high and low valuation boundaries based on ensemble tree variance.
- **Smart Deal Classification**: Categorizes predictions into **Good Deal**, **Fair Price**, or **Overpriced**.
- **Dark Glassmorphic UI**: Modern interface with responsive layout, dynamic brand/model dropdown auto-filtering, and micro-interactions.
- **Analytics Dashboard**: Interactive Chart.js graphs displaying valuation timelines, depreciation curves by year, and search/filterable history table.
- **Database Support**: Automatic schema initialization with parameter abstraction for both **SQLite** and **PostgreSQL**.
- **REST API & Data Export**: Includes JSON endpoints for metadata, predictions, history, and CSV download export.

---

## 🌐 Live Web Application Links

- 🏠 **Home Page**: [https://car-price-predictor-r4a7.onrender.com](https://car-price-predictor-r4a7.onrender.com)
- 🔮 **Predict Price**: [https://car-price-predictor-r4a7.onrender.com/predict](https://car-price-predictor-r4a7.onrender.com/predict)
- 📊 **Dashboard**: [https://car-price-predictor-r4a7.onrender.com/dashboard](https://car-price-predictor-r4a7.onrender.com/dashboard)
- ⚙️ **Metadata REST API**: [https://car-price-predictor-r4a7.onrender.com/api/meta](https://car-price-predictor-r4a7.onrender.com/api/meta)

---

## 📐 Architecture & Technology Stack

| Component | Technologies |
| :--- | :--- |
| **Backend Framework** | Flask, Python 3.11/3.14, Gunicorn |
| **Machine Learning** | scikit-learn (RandomForestRegressor, OneHotEncoder, StandardScaler), Joblib |
| **Data Processing** | Pandas, NumPy |
| **Database** | SQLite / PostgreSQL (psycopg2) |
| **Frontend** | HTML5, CSS3 (Custom Glassmorphism), Bootstrap 4.6, FontAwesome 6 |
| **Data Visualization**| Chart.js |
| **Cloud Deployment** | Render (Gunicorn + Python Blueprint) |

---

## 🚀 Quick Start Guide (Local Development)

### 1. Prerequisites
Ensure Python 3.8+ is installed on your system.

### 2. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/Abhi123ram1/car-price-predictor.git
cd car-price-predictor
pip install -r requirements.txt
```

### 3. Train Model
```bash
python train_model.py
```

### 4. Run Flask Server
```bash
python app.py
```
Open your browser at `http://localhost:5000`.

---

## 📡 REST API Documentation

### 1. Get Model Metadata & Form Options
`GET /api/meta`

**Response:**
```json
{
  "metrics": {
    "r2": 0.9652,
    "mae": 70429.43,
    "rmse": 150956.43,
    "sample_count": 8128
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
  "Year": 2021,
  "Kilometres_Driven": 25000,
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
  "predicted_price": 520000.0,
  "price_low": 485000.0,
  "price_high": 555000.0,
  "confidence": 94.5,
  "recommendation": "Good Deal",
  "currency": "INR (₹)",
  "timestamp": "2026-08-09 20:50:00"
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
