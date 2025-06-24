from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained pipeline
pipeline = joblib.load("credit_default_pipeline.pkl")

# Create Flask app
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>kimjerry AI lab</title>
        <meta charset="UTF-8">
        <style>
            body {
                margin: 0;
                padding: 0;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: flex-start;
                font-family: sans-serif;
            }
            .content {
                margin-top: 20vh;
                text-align: center;
            }
        </style>
    </head>
    <body>
        <div class="content">
            <h2>kimjerry AI lab</h2>
        </div>
    </body>
    </html>
    """, 200

@app.route("/predict", methods=["GET"])
def predict_get():
    return jsonify({
        "message": "Please send a POST request to this endpoint with a JSON body to get the prediction",
        "example_payload": {
            "status": "no_checking_account",
            "duration": 60,
            "credit_history": "critical_account_other_credits_existing",
            "purpose": "retraining",
            "amount": 10000,
            "savings": "unknown_no_savings_account",
            "employment_duration": "unemployed",
            "installment_rate": 6,
            "personal_status_sex": "male_single",
            "other_debtors": "guarantor",
            "present_residence": 1,
            "property": "unknown_no_property",
            "age": 25,
            "other_installment_plans": "bank",
            "housing": "rent",
            "number_credits": 5,
            "job": "unemployed_unskilled_non_resident",
            "people_liable": 6,
            "telephone": "yes",
            "foreign_worker": "yes"
        }
    })

@app.route("/predict", methods=["POST"])
def predict_post():
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
    app.run(host='0.0.0.0', port=5000, debug=True)