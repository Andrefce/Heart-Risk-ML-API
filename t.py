import pickle
import numpy as np
import joblib 

# Load the model
model = joblib.load("rf__model.pkl")

# Print feature importances (will print the importance of each feature in order)
print("Feature Importances:", model.feature_importances_)
