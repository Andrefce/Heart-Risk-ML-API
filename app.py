import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load the model
model = joblib.load("rf__model.pkl")

# Define the expected feature order based on model training
expected_features = [
    "General_Health", "Checkup", "Exercise", "Skin_Cancer", "Other_Cancer", 
    "Depression", "Diabetes", "Arthritis", "Sex", "Age_Category", "Height_(cm)", 
    "Weight_(kg)", "BMI", "Smoking_History", "Alcohol_Consumption", 
    "Fruit_Consumption", "Green_Vegetables_Consumption", "FriedPotato_Consumption"
]

# Categorical feature mappings
categorical_mappings = {
    "General_Health": {"Poor": 0, "Fair": 1, "Good": 2, "Very Good": 3, "Excellent": 4},
    "Checkup": {"Never": 0, "5 or more years ago": 1, "Within the past 5 years": 2, "Within the past year": 3, "Within the past 2 years": 4},
    "Exercise": {"No": 0, "Yes": 1},
    "Skin_Cancer": {"No": 0, "Yes": 1},
    "Other_Cancer": {"No": 0, "Yes": 1},
    "Depression": {"No": 0, "Yes": 1},
    "Diabetes": {"No": 0, "pre-diabetes or borderline diabetes": 1, "Yes": 2},
    "Arthritis": {"No": 0, "Yes": 1},
    "Sex": {"Female": 0, "Male": 1},
    "Age_Category": {
        "18-24": 0, "25-29": 1, "30-34": 2, "35-39": 3, "40-44": 4, "45-49": 5, "50-54": 6,
        "55-59": 7, "60-64": 8, "65-69": 9, "70-74": 10, "75-79": 11, "80+": 12
    },
    "Smoking_History": {"No": 0, "Yes": 1}
}

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        # Parse incoming JSON data
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON payload"}), 400

        # Check if all expected features are provided
        if not all(feature in data for feature in expected_features):
            return jsonify({"error": "Missing required features in the input"}), 400

        # Process categorical features by mapping them
        for feature, mapping in categorical_mappings.items():
            if feature in data and data[feature] in mapping:
                data[feature] = mapping[data[feature]]

        # Ensure the features are in the correct order (matching the order used during training)
        features = [data[feature] for feature in expected_features]

        # Create a DataFrame to maintain the expected feature order
        input_df = pd.DataFrame([features], columns=expected_features)

        # Make a prediction
        prediction = model.predict(input_df)

        # Return the prediction result
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
