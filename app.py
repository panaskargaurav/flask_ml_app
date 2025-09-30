import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

MODEL_PATH = "model.pkl"
ENCODER_PATH = "encoder_columns.pkl"  # Save columns after one-hot encoding

# ---------- Utility Functions ----------
def load_data(file):
    """Load CSV or Excel file into pandas DataFrame"""
    filename = file.filename
    if filename.endswith(".csv"):
        df = pd.read_csv(file)
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        df = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV or Excel.")
    return df

def save_model(model, encoder_columns):
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(encoder_columns, f)

def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError("Model not trained yet. Please train first.")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder_columns = pickle.load(f)
    return model, encoder_columns

def encode_features(df, encoder_columns=None):
    """One-hot encode 'area' column if exists"""
    if 'area' in df.columns:
        df = pd.get_dummies(df, columns=['area'])
    if encoder_columns is not None:
        # Add missing columns that exist in train but not in current df
        for col in encoder_columns:
            if col not in df.columns:
                df[col] = 0
        # Keep columns in same order as encoder_columns
        df = df[encoder_columns]
    return df

# ---------- API Routes ----------
@app.route("/train", methods=["POST"])
def train():
    try:
        file = request.files.get("file")
        target_column = request.form.get("target")

        if not file or not target_column:
            return jsonify({"error": "File and target column name are required"}), 400

        df = load_data(file)

        if target_column not in df.columns:
            return jsonify({"error": f"Target column '{target_column}' not found in dataset"}), 400

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical columns
        X_encoded = encode_features(X)
        encoder_columns = X_encoded.columns.tolist()

        X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        save_model(model, encoder_columns)

        return jsonify({"message": "Model trained successfully", "mse": mse, "r2": r2}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/test", methods=["POST"])
def test():
    try:
        file = request.files.get("file")
        target_column = request.form.get("target")

        if not file or not target_column:
            return jsonify({"error": "File and target column name are required"}), 400

        df = load_data(file)

        if target_column not in df.columns:
            return jsonify({"error": f"Target column '{target_column}' not found in dataset"}), 400

        model, encoder_columns = load_model()

        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_encoded = encode_features(X, encoder_columns)

        y_pred = model.predict(X_encoded)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        return jsonify({"message": "Model tested successfully", "mse": mse, "r2": r2}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model, encoder_columns = load_model()
        data = request.get_json()

        if not data:
            return jsonify({"error": "JSON input is required"}), 400

        df = pd.DataFrame([data])
        X_encoded = encode_features(df, encoder_columns)
        prediction = round(model.predict(X_encoded)[0], 2)  # Rounded to 2 decimals

        return jsonify({"prediction": prediction}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Run App ----------
if __name__ == "__main__":
    app.run(debug=True)
