import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import json
import pandas as pd
from datetime import datetime

app = Flask(__name__)


class DrunkardDetector:
    def __init__(self, model_paths):
        self.model_paths = model_paths
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_models()

    def load_models(self):
        """Load drunkard detection models"""
        try:
            # Load main model
            with open(self.model_paths['model'], 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                self.feature_names = model_data['feature_names']

            print("✓ Drunkard detection models loaded successfully")
            print(f"  Features: {len(self.feature_names)}")

        except Exception as e:
            print(f"Error loading models: {e}")
            self.create_fallback_model()

    def create_fallback_model(self):
        """Create fallback model if loading fails"""
        print("Creating fallback model...")

        # Create simple linear model
        from sklearn.linear_model import LinearRegression
        self.model = LinearRegression()

        # Create default scaler
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

        # Default features based on alcohol detection dataset
        self.feature_names = [
            'Alcohol_Level_ppm',
            'Heart_Rate_bpm',
            'Movement_Score',
            'Temperature_Celsius',
            'Humidity_Percent',
            'Ambient_Alcohol_ppm',
            'Hour',
            'DayOfWeek'
        ]

    def predict_drunkard_level(self, input_features):
        """Predict drunkard level based on input features"""
        try:
            # Prepare input
            input_df = pd.DataFrame([input_features])

            # Ensure all features are present
            missing_features = set(self.feature_names) - set(input_df.columns)
            for feature in missing_features:
                input_df[feature] = 0  # Default value

            input_df = input_df[self.feature_names]

            # Scale features
            input_scaled = self.scaler.transform(input_df)

            # Predict alcohol level
            prediction = self.model.predict(input_scaled)[0]

            # Determine drunkard level
            drunkard_level = self.classify_drunkard_level(prediction)

            return {
                'predicted_alcohol_level': float(prediction),
                'drunkard_level': drunkard_level['level'],
                'level_description': drunkard_level['description'],
                'risk_score': drunkard_level['risk_score'],
                'color': drunkard_level['color'],
                'recommendations': drunkard_level['recommendations'],
                'legal_limit_exceeded': prediction > 0.08,
                'confidence': self.calculate_confidence(prediction)
            }

        except Exception as e:
            print(f"Error in prediction: {e}")
            return self.fallback_prediction(input_features)

    def classify_drunkard_level(self, alcohol_level):
        """Classify drunkard level based on alcohol concentration"""

        if alcohol_level < 0.02:
            return {
                'level': 'SOBER',
                'description': 'No alcohol detected or minimal consumption',
                'risk_score': 0,
                'color': 'green',
                'recommendations': ['Safe to drive', 'No restrictions needed']
            }
        elif alcohol_level < 0.05:
            return {
                'level': 'MILD',
                'description': 'Minimal alcohol effects detectable',
                'risk_score': 25,
                'color': 'blue',
                'recommendations': ['Exercise caution', 'Avoid long drives']
            }
        elif alcohol_level < 0.08:
            return {
                'level': 'MODERATE',
                'description': 'Alcohol effects noticeable, approaching legal limit',
                'risk_score': 50,
                'color': 'orange',
                'recommendations': ['Should not drive', 'Consider alternative transport']
            }
        elif alcohol_level < 0.15:
            return {
                'level': 'HIGH',
                'description': 'Legally intoxicated, significant impairment',
                'risk_score': 75,
                'color': 'red',
                'recommendations': ['DO NOT DRIVE', 'Seek assistance', 'Report to authorities']
            }
        else:
            return {
                'level': 'SEVERE',
                'description': 'Severely intoxicated, dangerous impairment',
                'risk_score': 100,
                'color': 'darkred',
                'recommendations': ['MEDICAL ATTENTION NEEDED', 'Immediate intervention required',
                                    'Contact emergency services']
            }

    def calculate_confidence(self, alcohol_level):
        """Calculate prediction confidence"""
        # Higher confidence for extreme values
        if alcohol_level < 0.02 or alcohol_level > 0.15:
            return 0.95
        elif alcohol_level < 0.05 or alcohol_level > 0.08:
            return 0.85
        else:
            return 0.75

    def fallback_prediction(self, input_features):
        """Fallback prediction method"""
        alcohol_level = input_features.get('Alcohol_Level_ppm', 0.04)
        return {
            'predicted_alcohol_level': float(alcohol_level),
            'drunkard_level': 'MILD',
            'level_description': 'Estimated from input data',
            'risk_score': 30,
            'color': 'blue',
            'recommendations': ['Use with caution', 'Verify with breathalyzer'],
            'legal_limit_exceeded': alcohol_level > 0.08,
            'confidence': 0.7,
            'note': 'Using fallback prediction method'
        }


# Initialize detector
model_paths = {
    'model': '2models/alcohol_detection_model.pkl',
    'scaler': '2models/alcohol_detection_model_scaler.pkl'
}

detector = DrunkardDetector(model_paths)


@app.route('/drunkard_level')
def drunkard_level_page():
    """Render drunkard level prediction page"""
    return render_template('drunkard_level.html')


@app.route('/api/predict_drunkard', methods=['POST'])
def predict_drunkard():
    """API endpoint for drunkard level prediction"""
    try:
        data = request.get_json()

        # Required features
        required_features = {
            'Alcohol_Level_ppm': data.get('alcohol_level', 0.04),
            'Heart_Rate_bpm': data.get('heart_rate', 75),
            'Movement_Score': data.get('movement_score', 50),
            'Temperature_Celsius': data.get('temperature', 25),
            'Humidity_Percent': data.get('humidity', 50),
            'Ambient_Alcohol_ppm': data.get('ambient_alcohol', 0.01),
            'Hour': data.get('hour', datetime.now().hour),
            'DayOfWeek': data.get('day_of_week', datetime.now().weekday())
        }

        # Add additional features if provided
        for key in ['Driver_Age', 'Driver_Experience', 'Time_Since_Last_Meal']:
            if key in data:
                required_features[key] = data[key]

        # Make prediction
        result = detector.predict_drunkard_level(required_features)

        return jsonify({
            'success': True,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/api/drunkard_stats')
def get_drunkard_stats():
    """Get drunkard level statistics"""
    # This would typically come from a database
    stats = {
        'total_tests': 1250,
        'sober_count': 850,
        'mild_count': 250,
        'moderate_count': 100,
        'high_count': 40,
        'severe_count': 10,
        'legal_violations': 150,
        'avg_alcohol_level': 0.035,
        'highest_recorded': 0.28
    }

    return jsonify(stats)


if __name__ == '__main__':
    app.run(debug=True, port=5001)