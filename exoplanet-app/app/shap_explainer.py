#!/usr/bin/env python3
"""
SHAP Explainability Module for Exoplanet Classification
Provides interpretable explanations for model predictions using SHAP values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import shap
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class ExoplanetExplainer:
    """
    Provides SHAP-based explanations for exoplanet classification predictions.
    
    Uses SHAP (SHapley Additive exPlanations) to explain individual predictions
    and show which features most influenced the model's decision.
    """
    
    def __init__(self, model, feature_names: List[str], class_names: Optional[Dict] = None):
        """
        Initialize the explainer.
        
        Args:
            model: Trained model (XGBoost or LightGBM)
            feature_names: List of feature names in order
            class_names: Dictionary mapping class indices to names
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names or {0: 'False Positive', 1: 'Confirmed Planet', 2: 'Planet Candidate'}
        
        # Initialize SHAP explainer
        self.explainer_available = False
        
        if model is not None:
            try:
                if hasattr(model, 'predict_proba'):
                    self.explainer = shap.Explainer(model, feature_names=feature_names)
                else:
                    # Fallback for models without predict_proba
                    self.explainer = shap.Explainer(model.predict, feature_names=feature_names)
                
                self.explainer_available = True
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize SHAP explainer: {e}")
                self.explainer_available = False
        else:
            # Model is None - used for visualization only
            self.explainer = None
            self.explainer_available = False
    
    def explain_prediction(self, X: np.ndarray, max_features: int = 15) -> Dict:
        """
        Generate SHAP explanations for a single prediction.
        
        Args:
            X: Feature vector (1D or 2D array)
            max_features: Maximum number of features to show in explanation
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        if not self.explainer_available:
            return {'error': 'SHAP explainer not available'}
        
        try:
            # Ensure X is 2D
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            # Calculate SHAP values
            if self.explainer is None:
                return {'error': 'SHAP explainer not initialized'}
            
            shap_values = self.explainer(X)
            
            # Handle different SHAP value formats
            if hasattr(shap_values, 'values'):
                if shap_values.values.ndim == 3:  # Multi-class case
                    # Take the predicted class or average across classes
                    predicted_class = np.argmax(self.model.predict_proba(X)[0])
                    values = shap_values.values[0, :, predicted_class]
                else:
                    values = shap_values.values[0]
                base_value = shap_values.base_values
            else:
                values = shap_values[0] if isinstance(shap_values, list) else shap_values
                base_value = 0.0
            
            # Get feature contributions
            feature_contributions = []
            for i, (feature_name, shap_val, feature_val) in enumerate(zip(
                self.feature_names, values, X[0])):
                
                feature_contributions.append({
                    'feature': feature_name,
                    'value': float(feature_val),
                    'shap_value': float(shap_val),
                    'contribution': 'positive' if shap_val > 0 else 'negative',
                    'magnitude': abs(float(shap_val))
                })
            
            # Sort by absolute SHAP value (importance)
            feature_contributions.sort(key=lambda x: x['magnitude'], reverse=True)
            
            # Take top features
            top_features = feature_contributions[:max_features]
            
            return {
                'shap_values': values,
                'base_value': base_value,
                'feature_contributions': feature_contributions,
                'top_features': top_features,
                'total_positive_impact': sum(fc['shap_value'] for fc in feature_contributions if fc['shap_value'] > 0),
                'total_negative_impact': sum(fc['shap_value'] for fc in feature_contributions if fc['shap_value'] < 0)
            }
            
        except Exception as e:
            return {'error': f'SHAP explanation failed: {str(e)}'}
    
    def create_waterfall_plot(self, explanation: Dict, prediction_class: Optional[str] = None) -> matplotlib.figure.Figure:
        """
        Create a waterfall plot showing feature contributions.
        
        Args:
            explanation: SHAP explanation dictionary
            prediction_class: Name of predicted class
            
        Returns:
            Matplotlib figure
        """
        if 'error' in explanation:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error: {explanation['error']}", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        top_features = explanation['top_features'][:10]  # Show top 10
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for waterfall plot
        features = [f['feature'] for f in top_features]
        shap_values = [f['shap_value'] for f in top_features]
        
        # Create waterfall-style plot
        y_pos = np.arange(len(features))
        colors = ['green' if val > 0 else 'red' for val in shap_values]
        
        bars = ax.barh(y_pos, shap_values, color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels([self._clean_feature_name(f) for f in features])
        ax.set_xlabel('SHAP Value (Impact on Prediction)')
        
        title = f'Feature Contributions to Prediction'
        if prediction_class:
            title += f' ({prediction_class})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Add value labels on bars
        for bar, value in zip(bars, shap_values):
            width = bar.get_width()
            ax.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center')
        
        # Add vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add legend
        ax.text(0.02, 0.98, 'Green: Increases probability\nRed: Decreases probability', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def create_feature_importance_plot(self, explanation: Dict) -> matplotlib.figure.Figure:
        """
        Create a feature importance plot based on SHAP values.
        
        Args:
            explanation: SHAP explanation dictionary
            
        Returns:
            Matplotlib figure  
        """
        if 'error' in explanation:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f"Error: {explanation['error']}", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        top_features = explanation['top_features'][:15]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract data
        features = [self._clean_feature_name(f['feature']) for f in top_features]
        importances = [f['magnitude'] for f in top_features]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color='skyblue', alpha=0.8)
        
        # Customize plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Feature Importance (|SHAP Value|)')
        ax.set_title('Feature Importance for This Prediction', fontsize=14, fontweight='bold')
        
        # Add value labels
        for bar, importance in zip(bars, importances):
            width = bar.get_width()
            ax.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{importance:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        return fig
    
    def get_explanation_summary(self, explanation: Dict, input_data: Optional[Dict] = None) -> str:
        """
        Generate a text summary of the SHAP explanation.
        
        Args:
            explanation: SHAP explanation dictionary
            input_data: Original input data for context
            
        Returns:
            Text summary of the explanation
        """
        if 'error' in explanation:
            return f"Could not generate explanation: {explanation['error']}"
        
        top_features = explanation['top_features'][:5]
        
        summary = "**Key factors influencing this prediction:**\n\n"
        
        for i, feature in enumerate(top_features, 1):
            feature_name = self._clean_feature_name(feature['feature'])
            impact = "increases" if feature['shap_value'] > 0 else "decreases"
            strength = "strongly" if feature['magnitude'] > 0.1 else "moderately" if feature['magnitude'] > 0.05 else "slightly"
            
            value_context = ""
            if input_data and feature['feature'] in input_data:
                value_context = f" (value: {input_data[feature['feature']]:.2f})"
            
            summary += f"{i}. **{feature_name}**{value_context} {strength} {impact} the prediction\n"
        
        total_pos = explanation['total_positive_impact']
        total_neg = explanation['total_negative_impact']
        
        summary += f"\n**Overall impact balance:**\n"
        summary += f"- Positive contributions: {total_pos:.3f}\n"
        summary += f"- Negative contributions: {total_neg:.3f}\n"
        summary += f"- Net effect: {total_pos + total_neg:.3f}"
        
        return summary
    
    def _clean_feature_name(self, feature_name: str) -> str:
        """Clean up feature names for display."""
        # Remove technical suffixes
        name = feature_name.replace('_missing', ' (missing flag)')
        name = name.replace('_log10', ' (log10)')
        name = name.replace('_log', ' (log)')
        name = name.replace('_bin', ' (binned)')
        name = name.replace('_err', ' error')
        name = name.replace('_ratio', ' ratio')
        
        # Convert underscores to spaces and title case
        name = name.replace('_', ' ').title()
        
        # Fix common abbreviations
        replacements = {
            'Snr': 'SNR',
            'Teff': 'T_eff',
            'Logg': 'log g',
            'Sma': 'Semi-major axis',
            'Prad': 'Planet radius'
        }
        
        for old, new in replacements.items():
            name = name.replace(old, new)
        
        return name

# Test the explainer
if __name__ == "__main__":
    print("Testing SHAP Explainer...")
    
    # This would normally be run with actual model and data
    print("SHAP explainer module ready for integration.")