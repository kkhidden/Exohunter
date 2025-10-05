#!/usr/bin/env python3
"""
Exoplanet Model Inference Wrapper
Loads and uses trained XGBoost and LightGBM models for exoplanet classification.
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from feature_processor import ExoplanetFeatureProcessor
from shap_explainer import ExoplanetExplainer

class ExoplanetModelInference:
    """
    Wrapper class for loading and using trained exoplanet classification models.
    
    Supports both XGBoost and LightGBM models with ensemble predictions.
    """
    
    def __init__(self, model_dir: Optional[str] = None, schema_path: Optional[str] = None):
        """
        Initialize the model inference wrapper.
        
        Args:
            model_dir: Path to directory containing trained models
            schema_path: Path to schema.json file with feature definitions
        """
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).parent.parent / "model" / "models"
        self.schema_path = Path(schema_path) if schema_path else Path(__file__).parent.parent / "artifacts" / "schema.json"
        
        self.xgb_model = None
        self.lgb_model = None
        self.model_metadata = None
        self.feature_names = None
        self.class_names = None
        self.class_weights = None
        
        # Initialize feature processor
        self.feature_processor = ExoplanetFeatureProcessor()
        
        # Initialize explainers (will be set after models load)
        self.xgb_explainer = None
        self.lgb_explainer = None
        
        # Load schema and models
        self._load_schema()
        self._load_models()
        
        # Initialize explainers after models are loaded
        self._initialize_explainers()
    
    def _load_schema(self):
        """Load feature schema from JSON file."""
        try:
            with open(self.schema_path, 'r') as f:
                self.schema = json.load(f)
            
            print(f"✓ Schema loaded from {self.schema_path}")
            
            # Extract feature information
            self.numeric_features = self.schema.get('numeric_features', [])
            self.column_aliases = self.schema.get('column_aliases', {})
            self.target_classes = self.schema.get('target_classes_display_order', 
                                                ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
            
        except Exception as e:
            print(f"⚠️ Warning: Could not load schema from {self.schema_path}: {e}")
            self.schema = {}
            self.numeric_features = []
            self.column_aliases = {}
            self.target_classes = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
    
    def _load_models(self):
        """Load trained models from the models directory."""
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.model_dir}")
        
        # Find the most recent model files (by timestamp)
        xgb_files = list(self.model_dir.glob("xgboost_enhanced_*.pkl"))
        lgb_files = list(self.model_dir.glob("lightgbm_enhanced_*.pkl"))
        results_files = list(self.model_dir.glob("gradient_boosting_results_*.pkl"))
        
        if not xgb_files and not lgb_files:
            raise FileNotFoundError(f"No trained models found in {self.model_dir}")
        
        # Load most recent models (highest timestamp)
        if xgb_files:
            latest_xgb = max(xgb_files, key=lambda x: x.stem.split('_')[-1])
            try:
                with open(latest_xgb, 'rb') as f:
                    self.xgb_model = pickle.load(f)
                print(f"✓ XGBoost model loaded: {latest_xgb.name}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load XGBoost model: {e}")
        
        if lgb_files:
            latest_lgb = max(lgb_files, key=lambda x: x.stem.split('_')[-1])
            try:
                with open(latest_lgb, 'rb') as f:
                    self.lgb_model = pickle.load(f)
                print(f"✓ LightGBM model loaded: {latest_lgb.name}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load LightGBM model: {e}")
        
        # Load model metadata if available
        if results_files:
            latest_results = max(results_files, key=lambda x: x.stem.split('_')[-1])
            try:
                with open(latest_results, 'rb') as f:
                    self.model_metadata = pickle.load(f)
                
                self.feature_names = self.model_metadata.get('feature_names', [])
                self.class_names = self.model_metadata.get('class_names', self.target_classes)
                self.class_weights = self.model_metadata.get('class_weights', {})
                
                print(f"✓ Model metadata loaded: {latest_results.name}")
                print(f"  - Features: {len(self.feature_names)}")
                print(f"  - Classes: {self.class_names}")
                
            except Exception as e:
                print(f"⚠️ Warning: Could not load model metadata: {e}")
        
        # Fallback feature names if not available in metadata
        if not self.feature_names and (self.xgb_model or self.lgb_model):
            model = self.xgb_model if self.xgb_model else self.lgb_model
            if model and hasattr(model, 'feature_names_in_'):
                self.feature_names = list(model.feature_names_in_)
            elif model and hasattr(model, 'feature_name_'):
                self.feature_names = model.feature_name_
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models."""
        info = {
            'models_available': [],
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names,
            'class_names': self.class_names,
            'target_classes': self.target_classes
        }
        
        if self.xgb_model:
            info['models_available'].append('XGBoost')
        if self.lgb_model:
            info['models_available'].append('LightGBM')
        
        if self.model_metadata:
            info['model_metadata'] = {
                'timestamp': self.model_metadata.get('timestamp'),
                'data_shape': self.model_metadata.get('data_shape'),
                'target_f1': self.model_metadata.get('target_f1')
            }
        
        return info
    
    def preprocess_features(self, input_data: Dict[str, Union[float, int]]) -> np.ndarray:
        """
        Preprocess input features to match training data format.
        
        Args:
            input_data: Dictionary with feature names and values
            
        Returns:
            Preprocessed feature array ready for model prediction
        """
        # Use the feature processor to create the complete feature vector
        feature_vector = self.feature_processor.create_feature_vector(input_data)
        return feature_vector.reshape(1, -1)
    
    def _get_feature_default(self, feature_name: str) -> float:
        """Get default value for a feature from schema."""
        for feature_info in self.numeric_features:
            if (feature_info['name'] == feature_name or 
                feature_name in self.column_aliases.get(feature_info['name'], [])):
                return feature_info.get('default', 0.0)
        return 0.0
    
    def predict(self, input_data: Dict[str, Union[float, int]], 
                use_ensemble: bool = True) -> Dict:
        """
        Make prediction on input features.
        
        Args:
            input_data: Dictionary with feature names and values
            use_ensemble: Whether to use ensemble of models (if multiple available)
            
        Returns:
            Dictionary with predictions, probabilities, and metadata
        """
        if not self.xgb_model and not self.lgb_model:
            raise ValueError("No models are loaded for prediction")
        
        # Preprocess features
        X = self.preprocess_features(input_data)
        
        predictions = {}
        probabilities = {}
        
        # XGBoost predictions
        if self.xgb_model:
            try:
                xgb_pred = self.xgb_model.predict(X)[0]
                xgb_proba = self.xgb_model.predict_proba(X)[0]
                predictions['xgboost'] = xgb_pred
                probabilities['xgboost'] = xgb_proba
            except Exception as e:
                print(f"⚠️ Warning: XGBoost prediction failed: {e}")
        
        # LightGBM predictions
        if self.lgb_model:
            try:
                lgb_pred = self.lgb_model.predict(X)[0]
                lgb_proba = self.lgb_model.predict_proba(X)[0]
                predictions['lightgbm'] = lgb_pred
                probabilities['lightgbm'] = lgb_proba
            except Exception as e:
                print(f"⚠️ Warning: LightGBM prediction failed: {e}")
        
        # Ensemble prediction (average probabilities)
        ensemble_proba = None
        ensemble_pred = None
        
        if use_ensemble and len(probabilities) > 1:
            # Average probabilities across models
            all_probas = np.array(list(probabilities.values()))
            ensemble_proba = np.mean(all_probas, axis=0)
            ensemble_pred = np.argmax(ensemble_proba)
        elif probabilities:
            # Use single model result
            model_name = list(probabilities.keys())[0]
            ensemble_proba = probabilities[model_name]
            ensemble_pred = predictions[model_name]
        
        # Format results
        result = {
            'prediction': ensemble_pred,
            'prediction_label': self.class_names[ensemble_pred] if self.class_names else str(ensemble_pred),
            'confidence': float(np.max(ensemble_proba)) if ensemble_proba is not None else 0.0,
            'probabilities': {
                class_name: float(prob) for class_name, prob in 
                zip(self.class_names, ensemble_proba)
            } if ensemble_proba is not None and self.class_names else {},
            'individual_predictions': predictions,
            'individual_probabilities': {
                model: {self.class_names[i]: float(prob) for i, prob in enumerate(proba)} 
                for model, proba in probabilities.items()
            } if self.class_names else probabilities,
            'features_used': len(self.feature_names) if self.feature_names else 0,
            'models_used': list(predictions.keys())
        }
        
        return result
    
    def get_feature_importance(self, model_name: str = 'xgboost') -> Optional[Dict]:
        """
        Get feature importance from specified model.
        
        Args:
            model_name: 'xgboost' or 'lightgbm'
            
        Returns:
            Dictionary with feature names and importance scores
        """
        model = None
        if model_name == 'xgboost' and self.xgb_model:
            model = self.xgb_model
        elif model_name == 'lightgbm' and self.lgb_model:
            model = self.lgb_model
        
        if not model or not self.feature_names:
            return None
        
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return {
                    feature: float(importance) 
                    for feature, importance in zip(self.feature_names, importances)
                }
        except Exception as e:
            print(f"⚠️ Warning: Could not get feature importance: {e}")
        
        return None
    
    def _initialize_explainers(self):
        """Initialize SHAP explainers for loaded models."""
        try:
            if self.xgb_model and self.feature_names:
                self.xgb_explainer = ExoplanetExplainer(
                    self.xgb_model, 
                    self.feature_names, 
                    self.class_names or {}
                )
                print("✓ XGBoost SHAP explainer initialized")
                
            if self.lgb_model and self.feature_names:
                self.lgb_explainer = ExoplanetExplainer(
                    self.lgb_model,
                    self.feature_names,
                    self.class_names or {}
                )
                print("✓ LightGBM SHAP explainer initialized")
                
        except Exception as e:
            print(f"⚠️ Warning: Could not initialize SHAP explainers: {e}")
    
    def explain_prediction(self, input_data: Dict[str, Union[float, int]], 
                          model_name: str = 'xgboost') -> Optional[Dict]:
        """
        Generate SHAP explanation for a prediction.
        
        Args:
            input_data: Dictionary with feature names and values
            model_name: 'xgboost' or 'lightgbm'
            
        Returns:
            SHAP explanation dictionary or None if not available
        """
        # Get the appropriate explainer
        explainer = None
        if model_name == 'xgboost' and self.xgb_explainer:
            explainer = self.xgb_explainer
        elif model_name == 'lightgbm' and self.lgb_explainer:
            explainer = self.lgb_explainer
        
        if not explainer:
            return None
        
        # Preprocess features
        X = self.preprocess_features(input_data)
        
        # Generate explanation
        explanation = explainer.explain_prediction(X)
        
        # Add input data context
        if 'error' not in explanation:
            explanation['input_data'] = input_data
            explanation['model_used'] = model_name
        
        return explanation
    
    def predict_with_explanation(self, input_data: Dict[str, Union[float, int]], 
                                use_ensemble: bool = True) -> Dict:
        """
        Make prediction with SHAP explanations.
        
        Args:
            input_data: Dictionary with feature names and values
            use_ensemble: Whether to use ensemble of models
            
        Returns:
            Prediction results with explanations
        """
        # Get base prediction
        result = self.predict(input_data, use_ensemble)
        
        # Add explanations if available
        explanations = {}
        
        if self.xgb_explainer:
            xgb_explanation = self.explain_prediction(input_data, 'xgboost')
            if xgb_explanation:
                explanations['xgboost'] = xgb_explanation
        
        if self.lgb_explainer:
            lgb_explanation = self.explain_prediction(input_data, 'lightgbm')
            if lgb_explanation:
                explanations['lightgbm'] = lgb_explanation
        
        result['explanations'] = explanations
        return result

# Initialize global model instance
_model_instance = None

def get_model_instance(**kwargs) -> ExoplanetModelInference:
    """Get singleton model instance."""
    global _model_instance
    if _model_instance is None:
        _model_instance = ExoplanetModelInference(**kwargs)
    return _model_instance

def predict_exoplanet(input_data: Dict[str, Union[float, int]]) -> Dict:
    """
    Convenience function for making predictions.
    
    Args:
        input_data: Dictionary with feature names and values
        
    Returns:
        Prediction results dictionary
    """
    model = get_model_instance()
    return model.predict(input_data)

if __name__ == "__main__":
    # Test the model loader
    print("Testing Exoplanet Model Inference...")
    
    try:
        model = ExoplanetModelInference()
        info = model.get_model_info()
        
        print("\nModel Information:")
        print(f"Available models: {info['models_available']}")
        print(f"Feature count: {info['feature_count']}")
        print(f"Classes: {info['class_names']}")
        
        # Test prediction with sample data
        sample_data = {
            'orbital_period': 10.5,
            'transit_duration': 3.2,
            'planet_radius': 1.1,
            'snr': 15.0,
            'impact_parameter': 0.3,
            'stellar_teff': 5500,
            'stellar_logg': 4.4,
            'stellar_radius': 1.0
        }
        
        print(f"\nTesting prediction with sample data: {sample_data}")
        result = model.predict(sample_data)
        
        print(f"\nPrediction Results:")
        print(f"Prediction: {result['prediction_label']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print(f"Probabilities: {result['probabilities']}")
        
    except Exception as e:
        print(f"Error: {e}")