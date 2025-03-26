from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature order
model = joblib.load("crop_yield_model.pkl")
feature_order = joblib.load("feature_order.pkl")

@app.route('/')
def home():
    return "🌾 KrishiKisanAI API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from request
        data = request.get_json()
        df = pd.DataFrame([data])

        # Ensure correct feature order
        df = df.reindex(columns=feature_order, fill_value=0)

        # Make prediction
        prediction = model.predict(df)[0]

        return jsonify({"predicted_yield": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
