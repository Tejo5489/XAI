import xgboost as xgb
import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import firebase_admin
from firebase_admin import credentials, firestore
import os

class ClinicalAI:
    def __init__(self):
        self.model = None
        self.explainer = None
        self.feature_names = [
            'age', 'height', 'weight', 'heartRate', 
            'bloodPressure', 'oxygen', 'temperature', 
            'infectionMarker', 'painWeight', 'respWeight'
        ]
        self._initialize_model()

    def _initialize_model(self):
        """Simulates training on MIMIC-III data for demonstration."""
        # Generating synthetic clinical data for training
        np.random.seed(42)
        data_size = 1000
        X = pd.DataFrame(np.random.rand(data_size, len(self.feature_names)), columns=self.feature_names)
        # Create a synthetic target: Risk of Sepsis
        # Sepsis high if Oxygen is low AND Heart Rate is high
        y = (X['heartRate'] > 0.7) & (X['oxygen'] < 0.4) | (X['infectionMarker'] > 0.8)
        y = y.astype(int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        # Training the XGBoost Model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective='binary:logistic'
        )
        self.model.fit(X_train, y_train)
        
        # Initializing SHAP Explainer
        self.explainer = shap.TreeExplainer(self.model)

    def predict_with_xai(self, vitals_dict, symptoms_dict):
        """
        Calculates Risk and SHAP Values.
        Input: Dictionaries of patient data.
        Output: Risk probability and SHAP feature contributions.
        """
        # Mapping inputs to the feature vector
        input_data = pd.DataFrame([{
            'age': vitals_dict.get('age', 45) / 100,
            'height': vitals_dict.get('height', 170) / 250,
            'weight': vitals_dict.get('weight', 70) / 200,
            'heartRate': vitals_dict.get('heartRate', 80) / 200,
            'bloodPressure': vitals_dict.get('bloodPressure', 120) / 220,
            'oxygen': vitals_dict.get('oxygen', 98) / 100,
            'temperature': vitals_dict.get('temperature', 37) / 42,
            'infectionMarker': vitals_dict.get('infectionMarker', 1) / 20,
            'painWeight': 1.0 if symptoms_dict.get('pain') else 0.0,
            'respWeight': 1.0 if symptoms_dict.get('breathless') else 0.0
        }])

        # 1. XGBoost Prediction
        prob = self.model.predict_proba(input_data)[0][1]

        # 2. SHAP Explanation
        shap_values = self.explainer.shap_values(input_data)
        
        # Formatting for Frontend
        contributions = []
        for i, name in enumerate(self.feature_names):
            contributions.append({
                "feature": name,
                "phi": float(shap_values[0][i])
            })

        return {
            "probability": float(prob),
            "contributions": contributions,
            "base_value": float(self.explainer.expected_value)
        }

class CloudManager:
    def __init__(self, service_account_path='service_account.json'):
        if not firebase_admin._apps:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)
        self.db = firestore.client()

    def sync_assessment(self, user_id, app_id, assessment_data):
        """Writes audit log to Firestore following Rule 1."""
        path = f"artifacts/{app_id}/public/data/history"
        self.db.collection(path).add({
            **assessment_data,
            "userId": user_id,
            "server_timestamp": firestore.SERVER_TIMESTAMP
        })
