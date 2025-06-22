from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("credit_default_pipeline.pkl")

# Create Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Handle both single dict and list of dicts
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Input must be a JSON object or an array of objects."}), 400

        # Make predictions
        predictions = pipeline.predict(df).tolist()
        probabilities = pipeline.predict_proba(df).tolist()

        # Format results
        results = []
        for pred, proba in zip(predictions, probabilities):
            label = "Likely to pay back" if pred == 0 else "Likely to default (not pay back)"
            confidence_pct = proba[pred] * 100
            results.append({
                "confidence": f"{confidence_pct:.2f}%",
                "outcome": label
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)