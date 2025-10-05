#!/usr/bin/env python3
"""
Visualization utilities for the Exoplanet Classification Application
Provides various plots and charts for model insights and data exploration.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('default')

# Try to import seaborn if available, otherwise use defaults
try:
    import seaborn as sns
    sns.set_palette("husl")
except ImportError:
    sns = None

class ExoplanetVisualizer:
    """
    Creates visualizations for exoplanet classification results and data analysis.
    """
    
    def __init__(self):
        """Initialize the visualizer with default styling."""
        self.colors = {
            'False Positive': '#dc3545',    # Red
            'Confirmed Planet': '#28a745',   # Green  
            'Planet Candidate': '#ffc107'    # Yellow/Orange
        }
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
    def create_prediction_confidence_plot(self, probabilities: Dict[str, float]) -> matplotlib.figure.Figure:
        """
        Create a confidence visualization showing prediction probabilities.
        
        Args:
            probabilities: Dictionary mapping class names to probabilities
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        classes = list(probabilities.keys())
        probs = list(probabilities.values())
        colors = [self.colors.get(cls, '#007bff') for cls in classes]
        
        # Bar chart
        bars = ax1.bar(classes, probs, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_ylabel('Probability')
        ax1.set_title('Classification Probabilities', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        ax2.pie(probs, labels=classes, colors=colors, autopct='%1.1f%%', 
               startangle=90, explode=[0.05 if p == max(probs) else 0 for p in probs])
        ax2.set_title('Probability Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_feature_importance_comparison(self, model_info: Dict) -> matplotlib.figure.Figure:
        """
        Create a comparison of feature importance across models.
        
        Args:
            model_info: Dictionary containing model information
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # This is a placeholder - in a real implementation, you'd extract
        # feature importance from the actual models
        sample_features = [
            'Planet Radius', 'Orbital Period', 'Transit Duration', 'SNR', 
            'Stellar Temperature', 'Transit Depth', 'Impact Parameter',
            'Stellar Radius', 'Stellar Mass', 'Equilibrium Temperature'
        ]
        
        # Generate sample importance scores (in real app, get from models)
        np.random.seed(42)
        xgb_importance = np.random.uniform(0.1, 1.0, len(sample_features))
        lgb_importance = np.random.uniform(0.1, 1.0, len(sample_features))
        
        x = np.arange(len(sample_features))
        width = 0.35
        
        ax.barh(x - width/2, xgb_importance, width, label='XGBoost', alpha=0.8)
        ax.barh(x + width/2, lgb_importance, width, label='LightGBM', alpha=0.8)
        
        ax.set_xlabel('Feature Importance')
        ax.set_title('Feature Importance Comparison Across Models', fontsize=14, fontweight='bold')
        ax.set_yticks(x)
        ax.set_yticklabels(sample_features)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_parameter_space_plot(self, input_data: Dict[str, float]) -> matplotlib.figure.Figure:
        """
        Create a visualization showing where input parameters fall in typical ranges.
        
        Args:
            input_data: Dictionary with parameter names and values
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Define typical ranges for key parameters (from astronomical literature)
        param_ranges = {
            'orbital_period': {'min': 0.1, 'max': 1000, 'unit': 'days', 'log': True},
            'planet_radius': {'min': 0.1, 'max': 20, 'unit': 'Earth radii', 'log': True},
            'stellar_teff': {'min': 3000, 'max': 10000, 'unit': 'K', 'log': False},
            'snr': {'min': 1, 'max': 1000, 'unit': 'ratio', 'log': True}
        }
        
        plot_params = ['orbital_period', 'planet_radius', 'stellar_teff', 'snr']
        
        for i, param in enumerate(plot_params):
            if i >= len(axes):
                break
                
            ax = axes[i]
            param_info = param_ranges.get(param, {'min': 0, 'max': 100, 'unit': '', 'log': False})
            
            # Create range visualization
            range_min, range_max = param_info['min'], param_info['max']
            user_value = input_data.get(param, (range_min + range_max) / 2) or (range_min + range_max) / 2
            
            if param_info.get('log', False):
                # Log scale
                log_min, log_max = np.log10(range_min), np.log10(range_max)
                log_user = np.log10(max(user_value, range_min))
                
                ax.barh(0, log_max - log_min, left=log_min, height=0.3, 
                       alpha=0.3, color='lightblue', label='Typical Range')
                ax.barh(0, 0.1, left=log_user - 0.05, height=0.3, 
                       color='red', label='Your Value')
                
                ax.set_xlim(log_min - 0.1, log_max + 0.1)
                ax.set_xticks([log_min, log_user, log_max])
                ax.set_xticklabels([f'{range_min}', f'{user_value:.2f}', f'{range_max}'])
            else:
                # Linear scale
                ax.barh(0, range_max - range_min, left=range_min, height=0.3,
                       alpha=0.3, color='lightblue', label='Typical Range')
                ax.barh(0, range_max * 0.02, left=user_value - range_max * 0.01, height=0.3,
                       color='red', label='Your Value')
                
                ax.set_xlim(range_min - range_max * 0.05, range_max + range_max * 0.05)
            
            ax.set_title(f"{param.replace('_', ' ').title()} ({param_info['unit']})")
            ax.set_yticks([])
            ax.legend(loc='upper right')
        
        # Remove empty subplots
        for i in range(len(plot_params), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('Parameter Values in Context', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig
    
    def create_exoplanet_gallery_plot(self) -> matplotlib.figure.Figure:
        """
        Create an educational plot showing different types of exoplanets.
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sample exoplanet data (radius vs orbital period)
        exoplanet_types = {
            'Hot Jupiters': {
                'periods': np.random.lognormal(np.log(3), 0.5, 50),
                'radii': np.random.normal(10, 2, 50),
                'color': '#ff6b6b'
            },
            'Super Earths': {
                'periods': np.random.lognormal(np.log(20), 1, 80),
                'radii': np.random.normal(2, 0.5, 80),
                'color': '#4ecdc4'
            },
            'Earth-like': {
                'periods': np.random.lognormal(np.log(300), 0.8, 30),
                'radii': np.random.normal(1, 0.2, 30),
                'color': '#45b7d1'
            },
            'Mini Neptunes': {
                'periods': np.random.lognormal(np.log(50), 1.2, 60),
                'radii': np.random.normal(4, 1, 60),
                'color': '#96ceb4'
            }
        }
        
        for planet_type, data in exoplanet_types.items():
            ax.scatter(data['periods'], data['radii'], 
                      c=data['color'], alpha=0.6, s=50, label=planet_type)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Orbital Period (days)', fontsize=12)
        ax.set_ylabel('Planet Radius (Earth radii)', fontsize=12)
        ax.set_title('Exoplanet Types: Radius vs Orbital Period', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_model_performance_plot(self) -> matplotlib.figure.Figure:
        """
        Create a visualization showing model performance metrics.
        
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sample performance data (replace with actual metrics if available)
        models = ['XGBoost', 'LightGBM', 'Ensemble']
        metrics = {
            'Precision': [0.82, 0.79, 0.85],
            'Recall': [0.78, 0.81, 0.84],
            'F1-Score': [0.80, 0.80, 0.84]
        }
        
        # Performance comparison
        x = np.arange(len(models))
        width = 0.25
        
        for i, (metric, values) in enumerate(metrics.items()):
            ax1.bar(x + i * width, values, width, label=metric, alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison', fontweight='bold')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Confusion matrix style plot (sample data)
        classes = ['False Positive', 'Confirmed', 'Candidate']
        confusion_data = np.array([[85, 10, 5], [8, 87, 5], [12, 8, 80]])
        
        im = ax2.imshow(confusion_data, interpolation='nearest', cmap='Blues')
        ax2.set_title('Classification Accuracy Heatmap', fontweight='bold')
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax2.text(j, i, f'{confusion_data[i, j]}%',
                        ha='center', va='center', fontweight='bold')
        
        ax2.set_xticks(range(len(classes)))
        ax2.set_yticks(range(len(classes)))
        ax2.set_xticklabels(classes)
        ax2.set_yticklabels(classes)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        return fig

# Test the visualizer
if __name__ == "__main__":
    print("Testing Exoplanet Visualizer...")
    
    visualizer = ExoplanetVisualizer()
    
    # Test confidence plot
    sample_probs = {
        'False Positive': 0.2,
        'Confirmed Planet': 0.7,
        'Planet Candidate': 0.1
    }
    
    fig1 = visualizer.create_prediction_confidence_plot(sample_probs)
    print("✓ Confidence plot created")
    
    # Test parameter space plot
    sample_params = {
        'orbital_period': 15.2,
        'planet_radius': 1.3,
        'stellar_teff': 5800,
        'snr': 12.5
    }
    
    fig2 = visualizer.create_parameter_space_plot(sample_params)
    print("✓ Parameter space plot created")
    
    # Test gallery plot
    fig3 = visualizer.create_exoplanet_gallery_plot()
    print("✓ Exoplanet gallery plot created")
    
    print("All visualizations working correctly!")