"""
Flask web application for Used Car Price Prediction.

Routes:
- /             : Home page with model metrics & feature overview
- /predict      : Form to make predictions with interactive bounds & recommendations
- /dashboard    : View past predictions, charts & market analytics
- /api/meta     : Metadata REST API
- /api/predict  : JSON REST API for predictions
- /api/history  : JSON prediction history API
- /api/export-csv : Download predictions history as CSV
"""

from flask import Flask, render_template, request, jsonify, g, redirect, url_for, Response
import sqlite3
import os
import joblib
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import io
import csv

try:
    import psycopg2
    import psycopg2.extras
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

DB_PATH = "database.db"
MODEL_PATH = "model.pkl"
META_PATH = "model_meta.json"
DB_URL = os.environ.get("DATABASE_URL")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev_secret_key_car_predictor")

MODEL = None
META = {}

def load_model_and_meta():
    global MODEL, META
    if os.path.exists(MODEL_PATH):
        try:
            MODEL = joblib.load(MODEL_PATH)
            print("Model successfully loaded from", MODEL_PATH)
        except Exception as e:
            print("Error loading model:", e)

    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                META = json.load(f)
        except Exception as e:
            print("Error loading metadata:", e)

load_model_and_meta()

def get_db():
    db = getattr(g, "db", None)
    if db is not None:
        return db

    if DB_URL and HAS_POSTGRES:
        conn = psycopg2.connect(DB_URL)
        class PGWrapper:
            def __init__(self, conn):
                self.conn = conn
            def execute(self, sql, params=None):
                sql_pg = sql.replace("?", "%s")
                cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute(sql_pg, params or ())
                return cur
            def commit(self):
                self.conn.commit()
            def close(self):
                self.conn.close()

        db = g.db = PGWrapper(conn)
        return db
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        class SQLiteWrapper:
            def __init__(self, conn):
                self.conn = conn
            def execute(self, sql, params=None):
                cur = self.conn.cursor()
                if params:
                    cur.execute(sql, params)
                else:
                    cur.execute(sql)
                return cur
            def commit(self):
                self.conn.commit()
            def close(self):
                self.conn.close()

        db = g.db = SQLiteWrapper(conn)
        return db

def init_db():
    try:
        db = get_db()
        create_sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            timestamp TEXT,
            brand TEXT,
            model TEXT,
            year INTEGER,
            kilometres REAL,
            fuel_type TEXT,
            transmission TEXT,
            owner_count INTEGER,
            engine_cc REAL,
            mileage REAL,
            predicted_price REAL,
            price_low REAL,
            price_high REAL,
            confidence REAL,
            recommendation TEXT
        )
        """
        if not (DB_URL and HAS_POSTGRES):
            create_sql = create_sql.replace("id SERIAL PRIMARY KEY", "id INTEGER PRIMARY KEY AUTOINCREMENT")

        db.execute(create_sql)
        db.commit()
    except Exception as e:
        print("Database initialization notice:", e)

@app.before_request
def ensure_db_initialized():
    if not hasattr(app, "_db_initialized"):
        init_db()
        app._db_initialized = True

@app.teardown_appcontext
def close_db(exception):
    db = getattr(g, "db", None)
    if db is not None:
        try:
            db.close()
        except Exception:
            pass

def compute_price_range_and_confidence(model, X_df):
    pred = float(model.predict(X_df)[0])
    
    try:
        regressor = model.named_steps["regressor"]
        preprocessor = model.named_steps["preprocessor"]
        X_trans = preprocessor.transform(X_df)
        all_preds = np.array([est.predict(X_trans)[0] for est in regressor.estimators_])
        std = float(np.std(all_preds))
    except Exception:
        std = pred * 0.08

    margin = max(0.08 * pred, std * 1.5)
    low = max(10000, pred - margin)
    high = pred + margin

    if pred > 0:
        rel_uncertainty = std / pred
        confidence = max(60.0, min(98.5, (1.0 - rel_uncertainty * 1.8) * 100.0))
    else:
        confidence = 50.0

    return pred, low, high, confidence

def make_recommendation(predicted, low_market):
    if low_market <= 0:
        return "Fair Price"
    diff = (predicted - low_market) / low_market
    if diff <= -0.08:
        return "Good Deal"
    elif diff <= 0.12:
        return "Fair Price"
    else:
        return "Overpriced"

# --- Web Routes ---

@app.route("/")
def index():
    load_model_and_meta()
    return render_template("index.html", meta=META)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    load_model_and_meta()
    if request.method == "POST":
        try:
            form = request.form
            brand = form.get("brand", "").strip()
            car_model = form.get("model", "").strip()
            year = int(form.get("year", 2020))
            kilometres = float(form.get("kilometres", 50000))
            fuel_type = form.get("fuel_type", "Petrol").strip()
            transmission = form.get("transmission", "Manual").strip()
            owner_count = int(form.get("owner_count", 1))
            engine_cc = float(form.get("engine_cc", 1200))
            mileage = float(form.get("mileage", 18.0))

            if MODEL is None:
                return render_template("predict.html", meta=META, error="Model not loaded. Run train_model.py first.")

            X = pd.DataFrame([{
                "Brand": brand,
                "Model": car_model,
                "Year": year,
                "Kilometres_Driven": kilometres,
                "Fuel_Type": fuel_type,
                "Transmission": transmission,
                "Owner_Count": owner_count,
                "Engine_CC": engine_cc,
                "Mileage_kmpl": mileage
            }])

            pred, low, high, confidence = compute_price_range_and_confidence(MODEL, X)
            recommendation = make_recommendation(pred, low)

            now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

            db = get_db()
            db.execute(
                """INSERT INTO predictions 
                (timestamp, brand, model, year, kilometres, fuel_type, transmission, owner_count, engine_cc, mileage, predicted_price, price_low, price_high, confidence, recommendation) 
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now_str, brand, car_model, year, kilometres, fuel_type, transmission, owner_count, engine_cc, mileage, pred, low, high, confidence, recommendation),
            )
            db.commit()

            result = {
                "predicted_price": round(pred, 2),
                "price_low": round(low, 2),
                "price_high": round(high, 2),
                "confidence": round(confidence, 1),
                "recommendation": recommendation
            }

            input_data = X.iloc[0].to_dict()

            return render_template("predict.html", meta=META, result=result, input=input_data)

        except Exception as e:
            return render_template("predict.html", meta=META, error=str(e))

    return render_template("predict.html", meta=META)

@app.route("/dashboard")
def dashboard():
    load_model_and_meta()
    db = get_db()
    cur = db.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 100")
    rows = cur.fetchall()

    times = [r["timestamp"] for r in rows][::-1]
    prices = [r["predicted_price"] for r in rows][::-1]

    total_preds = len(rows)
    avg_price = sum(r["predicted_price"] for r in rows) / total_preds if total_preds > 0 else 0
    good_deals = sum(1 for r in rows if r["recommendation"] == "Good Deal")
    good_deals_pct = round((good_deals / total_preds * 100), 1) if total_preds > 0 else 0

    stats = {
        "total": total_preds,
        "avg_price": round(avg_price, 2),
        "good_deals_pct": good_deals_pct
    }

    price_by_year = META.get("price_by_year", [])
    price_by_brand = META.get("price_by_brand", [])

    return render_template(
        "dashboard.html", 
        rows=rows, 
        times=times, 
        prices=prices, 
        price_by_year=price_by_year,
        price_by_brand=price_by_brand,
        stats=stats,
        meta=META
    )

# --- REST API Endpoints ---

@app.route("/api/meta", methods=["GET"])
def api_meta():
    load_model_and_meta()
    return jsonify(META)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    load_model_and_meta()
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid or missing JSON payload"}), 400

    required = ["Brand", "Model", "Year", "Kilometres_Driven", "Fuel_Type", "Transmission", "Owner_Count", "Engine_CC", "Mileage_kmpl"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400

    if MODEL is None:
        return jsonify({"error": "Trained model unavailable. Run train_model.py."}), 500

    try:
        X = pd.DataFrame([{
            "Brand": str(data.get("Brand")),
            "Model": str(data.get("Model")),
            "Year": int(data.get("Year")),
            "Kilometres_Driven": float(data.get("Kilometres_Driven")),
            "Fuel_Type": str(data.get("Fuel_Type")),
            "Transmission": str(data.get("Transmission")),
            "Owner_Count": int(data.get("Owner_Count")),
            "Engine_CC": float(data.get("Engine_CC")),
            "Mileage_kmpl": float(data.get("Mileage_kmpl")),
        }])

        pred, low, high, confidence = compute_price_range_and_confidence(MODEL, X)
        recommendation = make_recommendation(pred, low)

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        db = get_db()
        db.execute(
            """INSERT INTO predictions 
            (timestamp, brand, model, year, kilometres, fuel_type, transmission, owner_count, engine_cc, mileage, predicted_price, price_low, price_high, confidence, recommendation) 
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now_str, data.get("Brand"), data.get("Model"), int(data.get("Year")), float(data.get("Kilometres_Driven")), data.get("Fuel_Type"), data.get("Transmission"), int(data.get("Owner_Count")), float(data.get("Engine_CC")), float(data.get("Mileage_kmpl")), pred, low, high, confidence, recommendation),
        )
        db.commit()

        return jsonify({
            "predicted_price": round(pred, 2),
            "price_low": round(low, 2),
            "price_high": round(high, 2),
            "confidence": round(confidence, 1),
            "recommendation": recommendation,
            "currency": "INR (₹)",
            "timestamp": now_str
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    db = get_db()
    cur = db.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 100")
    rows = cur.fetchall()
    result = [dict(r) for r in rows]
    return jsonify(result)

@app.route("/api/export-csv", methods=["GET"])
def api_export_csv():
    db = get_db()
    cur = db.execute("SELECT * FROM predictions ORDER BY id DESC")
    rows = cur.fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    
    headers = ["ID", "Timestamp", "Brand", "Model", "Year", "Kilometres", "Fuel_Type", "Transmission", "Owner_Count", "Engine_CC", "Mileage", "Predicted_Price", "Price_Low", "Price_High", "Confidence", "Recommendation"]
    writer.writerow(headers)

    for r in rows:
        writer.writerow([
            r["id"], r["timestamp"], r["brand"], r["model"], r["year"],
            r["kilometres"], r["fuel_type"], r["transmission"], r["owner_count"],
            r["engine_cc"], r["mileage"], r["predicted_price"], r["price_low"],
            r["price_high"], r["confidence"], r["recommendation"]
        ])

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=prediction_history.csv"}
    )

if __name__ == "__main__":
    with app.app_context():
        init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)

