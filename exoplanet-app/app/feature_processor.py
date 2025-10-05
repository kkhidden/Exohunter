#!/usr/bin/env python3
"""
Feature Preprocessing for Exoplanet Classification
Matches the preprocessing pipeline used during model training.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class ExoplanetFeatureProcessor:
    """
    Preprocesses input features to match the training data format.
    
    Handles feature engineering, missing value indicators, and transformations
    that were applied during model training.
    """
    
    def __init__(self):
        """Initialize the feature processor."""
        
        # Core features that the model expects (from training)
        self.core_features = [
            'period', 'duration', 'depth', 'impact_param', 'planet_radius',
            'semi_major_axis', 'equilibrium_temp', 'insolation', 'snr', 
            'num_transits', 'max_single_event', 'max_multi_event',
            'period_err', 'depth_err', 'duration_err', 'stellar_teff_err',
            'stellar_teff', 'stellar_logg', 'stellar_metallicity', 
            'stellar_radius', 'stellar_mass', 'kepler_mag', 'g_mag', 
            'r_mag', 'i_mag', 'quarters', 'radius_ratio', 'semi_major_axis_ratio'
        ]
        
        # Features that will have missing value indicators
        self.missing_indicator_features = [
            'depth', 'impact_param', 'planet_radius', 'semi_major_axis', 
            'equilibrium_temp', 'insolation', 'snr', 'num_transits', 
            'max_single_event', 'max_multi_event', 'period_err', 'depth_err', 
            'duration_err', 'stellar_teff_err', 'stellar_teff', 'stellar_logg', 
            'stellar_metallicity', 'stellar_radius', 'stellar_mass', 
            'kepler_mag', 'g_mag', 'r_mag', 'i_mag', 'quarters', 
            'radius_ratio', 'semi_major_axis_ratio'
        ]
        
        # Features for log10 transformation
        self.log10_features = [
            'period', 'depth', 'planet_radius', 'semi_major_axis', 
            'insolation', 'equilibrium_temp', 'stellar_mass'
        ]
        
        # Features for natural log transformation  
        self.log_features = ['period', 'depth', 'duration', 'snr']
        
        # Mapping from user-friendly names to internal feature names
        self.feature_aliases = {
            'orbital_period': 'period',
            'transit_duration': 'duration', 
            'transit_depth': 'depth',
            'impact_parameter': 'impact_param',
            'signal_to_noise_ratio': 'snr',
            'stellar_temperature': 'stellar_teff',
            'stellar_surface_gravity': 'stellar_logg',
            'stellar_metallicity': 'stellar_metallicity',
            'number_of_transits': 'num_transits',
            'orbital_semimajor_axis': 'semi_major_axis',
            'planet_equilibrium_temperature': 'equilibrium_temp',
            'stellar_insolation_flux': 'insolation'
        }
        
        # Default values for missing features (reasonable astronomical values)
        self.default_values = {
            'period': 10.0,           # days
            'duration': 3.0,          # hours  
            'depth': 100.0,           # ppm
            'impact_param': 0.5,      # 0-1
            'planet_radius': 1.0,     # Earth radii
            'semi_major_axis': 0.1,   # AU
            'equilibrium_temp': 300,  # K
            'insolation': 1.0,        # Earth flux
            'snr': 10.0,             # ratio
            'num_transits': 5,        # count
            'max_single_event': 10.0, # sigma
            'max_multi_event': 15.0,  # sigma
            'period_err': 0.1,        # days
            'depth_err': 10.0,        # ppm
            'duration_err': 0.1,      # hours
            'stellar_teff_err': 100,  # K
            'stellar_teff': 5500,     # K (Sun-like)
            'stellar_logg': 4.4,      # log(cm/s^2) (Sun-like)
            'stellar_metallicity': 0.0, # [Fe/H] (Solar)
            'stellar_radius': 1.0,    # Solar radii
            'stellar_mass': 1.0,      # Solar masses
            'kepler_mag': 12.0,       # magnitude
            'g_mag': 12.0,           # magnitude
            'r_mag': 12.0,           # magnitude  
            'i_mag': 12.0,           # magnitude
            'quarters': 10,           # number of quarters observed
            'radius_ratio': 0.01,     # Rp/R*
            'semi_major_axis_ratio': 10.0  # a/R*
        }
    
    def standardize_input(self, input_data: Dict[str, Union[float, int]]) -> Dict[str, float]:
        """
        Convert user input to standardized feature names and handle aliases.
        
        Args:
            input_data: Dictionary with user-provided feature values
            
        Returns:
            Dictionary with standardized feature names
        """
        standardized = {}
        
        for key, value in input_data.items():
            # Convert to lowercase and replace spaces/underscores 
            clean_key = key.lower().replace(' ', '_').replace('-', '_')
            
            # Check direct mapping
            if clean_key in self.core_features:
                standardized[clean_key] = float(value)
                continue
                
            # Check aliases
            if clean_key in self.feature_aliases:
                standard_name = self.feature_aliases[clean_key]
                standardized[standard_name] = float(value)
                continue
            
            # Try to match partial names (e.g., 'period' matches 'orbital_period')
            matched = False
            for alias, standard in self.feature_aliases.items():
                if clean_key in alias or alias in clean_key:
                    standardized[standard] = float(value)
                    matched = True
                    break
            
            if not matched:
                # Try direct feature name matching
                for feature in self.core_features:
                    if feature in clean_key or clean_key in feature:
                        standardized[feature] = float(value)
                        matched = True
                        break
            
            if not matched:
                print(f"⚠️ Warning: Could not map input '{key}' to known feature")
        
        return standardized
    
    def create_feature_vector(self, input_data: Dict[str, Union[float, int]]) -> np.ndarray:
        """
        Create complete feature vector matching training format.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Complete feature vector with all 69 features
        """
        # Standardize input names
        standardized = self.standardize_input(input_data)
        
        # Initialize with default values
        features = {}
        for feature in self.core_features:
            features[feature] = standardized.get(feature, self.default_values[feature])
        
        # Create missing value indicators
        missing_features = {}
        for feature in self.missing_indicator_features:
            missing_key = f"{feature}_missing"
            # 1 if missing (using default), 0 if provided
            missing_features[missing_key] = 0 if feature in standardized else 1
        
        # Create log10 transformed features
        log10_features = {}
        for feature in self.log10_features:
            if feature in features and features[feature] > 0:
                log10_features[f"{feature}_log10"] = np.log10(features[feature])
            else:
                log10_features[f"{feature}_log10"] = np.log10(self.default_values[feature])
        
        # Create natural log transformed features
        log_features = {}
        for feature in self.log_features:
            if feature in features and features[feature] > 0:
                log_features[f"{feature}_log"] = np.log(features[feature])
            else:
                log_features[f"{feature}_log"] = np.log(self.default_values[feature])
        
        # Create derived ratio features
        derived_features = {}
        if 'period' in features and 'duration' in features:
            derived_features['period_duration_ratio'] = features['period'] / (features['duration'] / 24.0)  # Convert hours to days
        else:
            derived_features['period_duration_ratio'] = 80.0  # Typical value
            
        if 'depth' in features and 'snr' in features:
            derived_features['depth_snr_ratio'] = features['depth'] / features['snr']
        else:
            derived_features['depth_snr_ratio'] = 10.0  # Typical value
        
        # Create binned features (simple quantile-based bins)
        binned_features = {}
        
        # Period bins (based on typical exoplanet periods)
        period_val = features.get('period', self.default_values['period'])
        if period_val < 1:
            binned_features['period_bin'] = 0
        elif period_val < 10:
            binned_features['period_bin'] = 1  
        elif period_val < 100:
            binned_features['period_bin'] = 2
        else:
            binned_features['period_bin'] = 3
        
        # Depth bins (based on typical transit depths in ppm)
        depth_val = features.get('depth', self.default_values['depth'])
        if depth_val < 50:
            binned_features['depth_bin'] = 0
        elif depth_val < 500:
            binned_features['depth_bin'] = 1
        elif depth_val < 5000:
            binned_features['depth_bin'] = 2
        else:
            binned_features['depth_bin'] = 3
        
        # Combine all features in the correct order
        all_features = {}
        
        # Core features (0-27)
        for feature in self.core_features:
            all_features[feature] = features[feature]
        
        # Missing indicators (28-53) 
        for feature in self.missing_indicator_features:
            missing_key = f"{feature}_missing"
            all_features[missing_key] = missing_features[missing_key]
        
        # Log10 features (54-60)
        for feature in self.log10_features:
            log10_key = f"{feature}_log10"
            all_features[log10_key] = log10_features[log10_key]
        
        # Derived ratios (61-62)
        all_features['period_duration_ratio'] = derived_features['period_duration_ratio']
        all_features['depth_snr_ratio'] = derived_features['depth_snr_ratio']
        
        # Natural log features (63-66)
        for feature in self.log_features:
            log_key = f"{feature}_log"  
            all_features[log_key] = log_features[log_key]
        
        # Binned features (67-68)
        all_features['period_bin'] = binned_features['period_bin']
        all_features['depth_bin'] = binned_features['depth_bin']
        
        # Convert to ordered numpy array (must match training order exactly)
        expected_order = [
            'period', 'duration', 'depth', 'impact_param', 'planet_radius',
            'semi_major_axis', 'equilibrium_temp', 'insolation', 'snr', 
            'num_transits', 'max_single_event', 'max_multi_event',
            'period_err', 'depth_err', 'duration_err', 'stellar_teff_err',
            'stellar_teff', 'stellar_logg', 'stellar_metallicity', 
            'stellar_radius', 'stellar_mass', 'kepler_mag', 'g_mag', 
            'r_mag', 'i_mag', 'quarters', 'radius_ratio', 'semi_major_axis_ratio',
            
            # Missing indicators
            'depth_missing', 'impact_param_missing', 'planet_radius_missing',
            'semi_major_axis_missing', 'equilibrium_temp_missing', 
            'insolation_missing', 'snr_missing', 'num_transits_missing',
            'max_single_event_missing', 'max_multi_event_missing', 
            'period_err_missing', 'depth_err_missing', 'duration_err_missing',
            'stellar_teff_err_missing', 'stellar_teff_missing', 
            'stellar_logg_missing', 'stellar_metallicity_missing',
            'stellar_radius_missing', 'stellar_mass_missing', 
            'kepler_mag_missing', 'g_mag_missing', 'r_mag_missing',
            'i_mag_missing', 'quarters_missing', 'radius_ratio_missing',
            'semi_major_axis_ratio_missing',
            
            # Log10 features
            'period_log10', 'depth_log10', 'planet_radius_log10',
            'semi_major_axis_log10', 'insolation_log10', 'equilibrium_temp_log10',
            'stellar_mass_log10',
            
            # Derived ratios
            'period_duration_ratio', 'depth_snr_ratio',
            
            # Natural log features  
            'period_log', 'depth_log', 'duration_log', 'snr_log',
            
            # Binned features
            'period_bin', 'depth_bin'
        ]
        
        # Create feature vector in correct order
        feature_vector = []
        for feature_name in expected_order:
            if feature_name in all_features:
                feature_vector.append(all_features[feature_name])
            else:
                print(f"⚠️ Warning: Missing feature {feature_name}, using 0")
                feature_vector.append(0.0)
        
        return np.array(feature_vector, dtype=np.float64)
    
    def get_feature_summary(self, input_data: Dict[str, Union[float, int]]) -> Dict:
        """
        Get summary of processed features for debugging/display.
        
        Args:
            input_data: Dictionary with feature values
            
        Returns:
            Summary dictionary with processing info
        """
        standardized = self.standardize_input(input_data)
        
        summary = {
            'input_features': len(input_data),
            'mapped_features': len(standardized),
            'missing_features': [],
            'provided_features': list(standardized.keys()),
            'using_defaults': []
        }
        
        for feature in self.core_features:
            if feature not in standardized:
                summary['missing_features'].append(feature)
                summary['using_defaults'].append(f"{feature} = {self.default_values[feature]}")
        
        return summary

# Test the feature processor
if __name__ == "__main__":
    print("Testing Exoplanet Feature Processor...")
    
    processor = ExoplanetFeatureProcessor()
    
    # Test with sample data
    sample_data = {
        'orbital_period': 15.2,
        'transit_duration': 4.1, 
        'planet_radius': 1.3,
        'signal_to_noise_ratio': 12.5,
        'stellar_temperature': 5800,
        'stellar_surface_gravity': 4.2,
        'stellar_radius': 1.1
    }
    
    print(f"\nInput data: {sample_data}")
    
    # Get feature summary
    summary = processor.get_feature_summary(sample_data)
    print(f"\nFeature Summary:")
    print(f"  Input features: {summary['input_features']}")
    print(f"  Mapped features: {summary['mapped_features']}")
    print(f"  Provided: {summary['provided_features']}")
    print(f"  Missing (using defaults): {len(summary['missing_features'])}")
    
    # Create feature vector
    feature_vector = processor.create_feature_vector(sample_data)
    print(f"\nGenerated feature vector:")
    print(f"  Shape: {feature_vector.shape}")
    print(f"  First 10 values: {feature_vector[:10]}")
    print(f"  All values finite: {np.isfinite(feature_vector).all()}")