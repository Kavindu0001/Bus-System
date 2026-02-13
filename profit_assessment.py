import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class ProfitAssessment:
    def __init__(self, models_dir='3models/'):
        self.models_dir = models_dir
        self.loaded_models = {}
        self.loaded_data = {}
        self.load_all_models()

    def load_all_models(self):
        """Load all profit assessment models"""
        print("Loading profit assessment models...")

        try:
            # Load feature names
            feature_path = os.path.join(self.models_dir, 'feature_names.pkl')
            if os.path.exists(feature_path):
                with open(feature_path, 'rb') as f:
                    self.loaded_models['feature_names'] = pickle.load(f)
                print(f"✓ Loaded feature names: {len(self.loaded_models['feature_names'])} features")

            # Load main model
            model_path = os.path.join(self.models_dir, 'model.pkl')
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.loaded_models['model'] = pickle.load(f)
                print("✓ Loaded main model")

            # Load performance metrics
            metrics_path = os.path.join(self.models_dir, 'performance_metrics.json')
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.loaded_models['performance'] = json.load(f)
                print("✓ Loaded performance metrics")

            # Load income prediction report
            report_path = os.path.join(self.models_dir, 'income_prediction_report.json')
            if os.path.exists(report_path):
                with open(report_path, 'r') as f:
                    self.loaded_data['income_report'] = json.load(f)
                print("✓ Loaded income prediction report")

        except Exception as e:
            print(f"Error loading models: {e}")

    def assess_profit(self, input_data):
        """Assess profit based on input parameters"""
        try:
            # Prepare input
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
            else:
                df = input_data.copy()

            # Get model components
            model_components = self.loaded_models.get('model')
            if not model_components:
                return self.generate_fallback_assessment(input_data)

            model = model_components['model']
            feature_names = model_components['feature_names']
            X_scaler = model_components['X_scaler']
            y_scaler = model_components['y_scaler']

            # Ensure all features are present
            for feature in feature_names:
                if feature not in df.columns:
                    if feature in ['daily_collection', 'total_distance', 'fuel_cost_per_l']:
                        df[feature] = 0
                    else:
                        df[feature] = 0.0

            df = df[feature_names]

            # Scale and predict
            X_scaled = X_scaler.transform(df)
            prediction_scaled = model.predict(X_scaled)
            predicted_profit = y_scaler.inverse_transform(prediction_scaled.reshape(-1, 1)).flatten()[0]

            # Generate assessment
            assessment = self.generate_profit_assessment(predicted_profit, input_data)

            return assessment

        except Exception as e:
            print(f"Error in profit assessment: {e}")
            return self.generate_fallback_assessment(input_data)

    def generate_profit_assessment(self, profit, input_data):
        """Generate comprehensive profit assessment"""

        # Determine profit status
        if profit >= 0:
            profit_status = "Profitable"
            status_color = "green"
            status_icon = "✅"
            recommendation_level = "GOOD"
        else:
            profit_status = "Loss Making"
            status_color = "red"
            status_icon = "⚠️"
            recommendation_level = "NEEDS IMPROVEMENT"

        # Calculate key metrics
        daily_collection = input_data.get('daily_collection', 175000)
        total_distance = input_data.get('total_distance', 400)

        revenue_per_km = daily_collection / total_distance if total_distance > 0 else 0
        profit_margin = (profit / daily_collection * 100) if daily_collection > 0 else 0

        # Generate recommendations
        recommendations = self.generate_recommendations(profit, input_data)

        # Generate risk assessment
        risk_assessment = self.assess_risk(profit, input_data)

        # Prepare assessment report
        assessment = {
            'profit': float(profit),
            'profit_status': profit_status,
            'status_color': status_color,
            'status_icon': status_icon,
            'recommendation_level': recommendation_level,
            'key_metrics': {
                'revenue_per_km': float(revenue_per_km),
                'profit_margin': float(profit_margin),
                'daily_collection': float(daily_collection),
                'total_distance': float(total_distance)
            },
            'recommendations': recommendations,
            'risk_assessment': risk_assessment,
            'temporal_predictions': self.predict_temporal_profits(profit),
            'generated_at': datetime.now().isoformat()
        }

        return assessment

    def generate_recommendations(self, profit, input_data):
        """Generate personalized recommendations"""
        recommendations = []

        if profit < 0:
            recommendations.append({
                'priority': 'HIGH',
                'title': 'Reduce Operational Costs',
                'description': 'Current operations are loss-making. Consider optimizing fuel consumption and reducing maintenance costs.',
                'action': 'Review fuel efficiency and negotiate maintenance contracts'
            })

        fuel_cost = input_data.get('fuel_cost_per_l', 360.5)
        if fuel_cost > 350:
            recommendations.append({
                'priority': 'MEDIUM',
                'title': 'Optimize Fuel Costs',
                'description': f'Fuel cost (LKR {fuel_cost}/liter) is above optimal range.',
                'action': 'Consider bulk purchasing or alternative fuel stations'
            })

        bus_age = input_data.get('bus_age_years', 7)
        if bus_age > 10:
            recommendations.append({
                'priority': 'MEDIUM',
                'title': 'Vehicle Maintenance',
                'description': f'Bus age ({bus_age} years) may be affecting operational efficiency.',
                'action': 'Schedule comprehensive maintenance check'
            })

        total_distance = input_data.get('total_distance', 400)
        if total_distance > 300:
            recommendations.append({
                'priority': 'LOW',
                'title': 'Route Optimization',
                'description': 'Long-distance operations detected. Could benefit from route optimization.',
                'action': 'Analyze route efficiency and consider route balancing'
            })

        return recommendations

    def assess_risk(self, profit, input_data):
        """Assess operational risk"""

        risk_factors = []
        risk_score = 0

        # Fuel cost risk
        fuel_cost = input_data.get('fuel_cost_per_l', 360.5)
        if fuel_cost > 370:
            risk_factors.append('High fuel costs')
            risk_score += 30

        # Bus age risk
        bus_age = input_data.get('bus_age_years', 7)
        if bus_age > 10:
            risk_factors.append('Aging vehicle')
            risk_score += 25

        # Distance risk
        total_distance = input_data.get('total_distance', 400)
        if total_distance > 350:
            risk_factors.append('Long-distance operations')
            risk_score += 20

        # Profit risk
        if profit < -100000:
            risk_factors.append('Significant losses')
            risk_score += 50
        elif profit < 0:
            risk_factors.append('Operating at loss')
            risk_score += 30

        # Determine risk level
        if risk_score >= 70:
            risk_level = 'HIGH'
            color = 'red'
        elif risk_score >= 40:
            risk_level = 'MEDIUM'
            color = 'orange'
        else:
            risk_level = 'LOW'
            color = 'green'

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'color': color
        }

    def predict_temporal_profits(self, daily_profit):
        """Predict profits for different time periods"""

        periods = {
            'daily': 1,
            'weekly': 7,
            'monthly': 30,
            'quarterly': 90,
            'annually': 365
        }

        predictions = {}
        for period, multiplier in periods.items():
            base_profit = daily_profit * multiplier

            # Apply growth for longer periods
            if period != 'daily':
                growth_rate = 0.05  # 5% growth
                growth_factor = (1 + growth_rate) ** (multiplier / 365)
                projected = base_profit * growth_factor
            else:
                projected = base_profit

            predictions[period] = {
                'base': float(base_profit),
                'projected': float(projected),
                'period_name': period.capitalize()
            }

        return predictions

    def generate_fallback_assessment(self, input_data):
        """Generate fallback assessment when models are not available"""
        print("Using fallback assessment (models not loaded)")

        # Simple calculation
        daily_collection = input_data.get('daily_collection', 175000)
        total_distance = input_data.get('total_distance', 400)
        fuel_cost = input_data.get('fuel_cost_per_l', 360.5)
        trips_count = input_data.get('trips_count', 3)

        # Simplified profit calculation
        fuel_consumption = total_distance / 3.5
        fuel_cost_total = fuel_consumption * fuel_cost
        operational_cost = fuel_cost_total + 750 + 2200 + 1650  # Maintenance + driver + conductor
        profit = daily_collection - operational_cost

        return self.generate_profit_assessment(profit, input_data)

    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'models_loaded': len(self.loaded_models) > 0,
            'model_count': len(self.loaded_models),
            'has_performance_metrics': 'performance' in self.loaded_models,
            'has_income_report': 'income_report' in self.loaded_data,
            'loaded_at': datetime.now().isoformat()
        }

        if 'performance' in self.loaded_models:
            perf = self.loaded_models['performance']
            info['model_performance'] = {
                'test_r2': perf.get('Test R2', 'N/A'),
                'test_rmse': perf.get('Test RMSE', 'N/A'),
                'test_mae': perf.get('Test MAE', 'N/A')
            }

        return info


# Create global instance
profit_assessor = ProfitAssessment()

if __name__ == "__main__":
    # Test the profit assessor
    test_input = {
        'daily_collection': 180000,
        'total_distance': 350,
        'fuel_cost_per_l': 360.5,
        'trips_count': 3,
        'maintenance_cost': 750,
        'driver_salary': 2200,
        'conductor_salary': 1650,
        'bus_age_years': 7
    }

    result = profit_assessor.assess_profit(test_input)
    print("Profit Assessment Result:")
    print(f"Predicted Profit: LKR {result['profit']:,.2f}")
    print(f"Status: {result['profit_status']}")
    print(f"Recommendation Level: {result['recommendation_level']}")
    print(f"Risk Level: {result['risk_assessment']['risk_level']}")