import os
import numpy as np
import pandas as pd
import joblib
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ======================
#  CONFIG
# ======================
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ACCURACIES_PATH = "model_accuracies.json"

model = None
scaler = None
model_accuracies = {}


# ======================
#  LOAD MODEL
# ======================
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print("Model loaded using joblib")
        except Exception:
            with open(MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            print("Model loaded using pickle")
    else:
        print("MODEL NOT FOUND")


# ======================
#  LOAD SCALER
# ======================
# ======================
#  LOAD ACCURACIES
# ======================
def load_accuracies():
    global model_accuracies
    if os.path.exists(ACCURACIES_PATH):
        with open(ACCURACIES_PATH, 'r') as f:
            model_accuracies = pd.read_json(f)
        print("Model accuracies loaded")
    else:
        print("MODEL ACCURACIES NOT FOUND")

def load_scaler():
    global scaler
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("Scaler loaded")
    else:
        print("SCALER NOT FOUND — creating new scaler")
        try:
            from sklearn.preprocessing import StandardScaler
            df = pd.read_csv("Cardio_cleaned.csv")

            feature = [
                'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                'cholesterol', 'gluc', 'smoke', 'alco',
                'active', 'Age_Year', 'bmi', 'pulse_pressure'
            ]

            X = df[feature]
            scaler = StandardScaler()
            scaler.fit(X)
            joblib.dump(scaler, SCALER_PATH)
            print("Scaler created")
        except Exception as e:
            print("Scaler create failed:", e)
            scaler = None


load_model()
load_scaler()
load_accuracies()


# ======================
#  ROUTES (HTML PAGES)
# ======================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")


@app.route("/predict_ui")
def predict_ui():
    return render_template("predict.html")


@app.route("/result")
def result():
    return render_template("result.html")

@app.route("/accuracies")
def accuracies():
    return render_template("accuracy.html", accuracies=model_accuracies.to_dict(orient='records'))


# ======================
#  PREDICTION API
# ======================
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "MODEL NOT LOADED"}), 500
    if scaler is None:
        return jsonify({"error": "SCALER NOT LOADED"}), 500

    try:
        data = request.form

        gender = int(data.get('gender', 0))
        height = float(data.get('height', 170))
        weight = float(data.get('weight', 70))
        ap_hi = float(data.get('ap_hi', 120))
        ap_lo = float(data.get('ap_lo', 80))
        cholesterol = int(data.get('cholesterol', 1))
        gluc = int(data.get('gluc', 1))
        smoke = int(data.get('smoke', 0))
        alco = int(data.get('alco', 0))
        active = int(data.get('active', 1))
        age_year = float(data.get('Age_Year', 45))

        # BMI
        bmi = weight / ((height / 100) ** 2)

        # Pulse pressure
        pulse_pressure = ap_hi - ap_lo

        features = np.array([[gender, height, weight, ap_hi, ap_lo,
                              cholesterol, gluc, smoke, alco,
                              active, age_year, bmi, pulse_pressure]],
                            dtype=np.float64)

        # Scale
        features_scaled = scaler.transform(features)

        # Predict
        prediction = int(model.predict(features_scaled)[0])

        # Probability if available
        probability = None
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(features_scaled)[0][1]
            probability = round(float(p) * 100, 1)

        return jsonify({
            "success": True,
            "prediction": prediction,
            "probability": probability
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400


# ======================
#  RUN LOCAL ONLY
# ======================
if __name__ == "__main__":
    app.run()
