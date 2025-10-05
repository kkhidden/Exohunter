#!/usr/bin/env python3
"""
NASA Exoplanet Dataset Loader - Phase 1
Kepler Objects of Interest (KOI) Dataset Processing

This module provides clean, versioned dataset loading with preprocessing
for the Kepler exoplanet classification task.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KeplerDatasetLoader:
    """
    Kepler Objects of Interest (KOI) dataset loader and preprocessor.
    
    Handles data cleaning, feature engineering, and preprocessing for
    exoplanet classification with proper label leakage prevention.
    """
    
    def __init__(self, data_path: str = "/Users/kkgogada/Code/NASASAC2025/KOI.csv"):
        """
        Initialize the dataset loader.
        
        Args:
            data_path: Path to the KOI.csv file
        """
        self.data_path = Path(data_path)
        self.cache_path = Path("/Users/kkgogada/Code/NASASAC2025/dataset/kepler_processed.parquet")
        self.label_encoder = LabelEncoder()
        
        # Define feature mapping for minimum required features
        self.feature_mapping = {
            # Orbital characteristics
            'period': 'koi_period',
            'duration': 'koi_duration', 
            'depth': 'koi_depth',
            'impact_param': 'koi_impact',
            
            # Planet characteristics
            'planet_radius': 'koi_prad',  # Planet radius in Earth radii
            'semi_major_axis': 'koi_sma', # Semi-major axis
            'equilibrium_temp': 'koi_teq', # Equilibrium temperature
            'insolation': 'koi_insol',    # Insolation flux
            
            # Signal quality
            'snr': 'koi_model_snr',       # Signal-to-noise ratio
            'num_transits': 'koi_num_transits',  # Number of transits
            'max_single_event': 'koi_max_sngle_ev', # Maximum single event
            'max_multi_event': 'koi_max_mult_ev',   # Maximum multiple event
            
            # Crowding metrics and model quality (already included above as max_single_event, max_multi_event)
            # Note: koi_model_chisq exists but is completely empty in dataset
            
            # Measurement uncertainties (quality indicators)
            'period_err': 'koi_period_err1',       # Period uncertainty
            'depth_err': 'koi_depth_err1',         # Depth uncertainty  
            'duration_err': 'koi_duration_err1',   # Duration uncertainty
            'stellar_teff_err': 'koi_steff_err1',  # Stellar temperature uncertainty
            
            # Stellar characteristics  
            'stellar_teff': 'koi_steff',   # Stellar effective temperature
            'stellar_logg': 'koi_slogg',   # Stellar surface gravity (log g)
            'stellar_metallicity': 'koi_smet', # Stellar metallicity [Fe/H]
            'stellar_radius': 'koi_srad',  # Stellar radius
            'stellar_mass': 'koi_smass',   # Stellar mass
            
            # Photometry and magnitudes
            'kepler_mag': 'koi_kepmag',    # Kepler magnitude
            'g_mag': 'koi_gmag',           # g-band magnitude
            'r_mag': 'koi_rmag',           # r-band magnitude
            'i_mag': 'koi_imag',           # i-band magnitude
            
            # Data quality indicators
            'quarters': 'koi_quarters',    # Kepler quarters observed
            
            # Additional quality metrics
            'radius_ratio': 'koi_ror',     # Planet-to-star radius ratio (Rp/R*)
            'semi_major_axis_ratio': 'koi_dor', # Semi-major axis to stellar radius ratio (a/R*)
        }
        
        # Columns that leak the target label (must be excluded)
        self.label_leak_columns = {
            'koi_disposition',      # Archive disposition (target)
            'koi_pdisposition',     # Kepler pipeline disposition  
            'koi_score',            # Disposition score
            'koi_fpflag_nt',        # Not transit-like flag
            'koi_fpflag_ss',        # Stellar eclipse flag
            'koi_fpflag_co',        # Centroid offset flag
            'koi_fpflag_ec',        # Ephemeris contamination flag
            'koi_disp_prov',        # Disposition provenance
            'koi_vet_stat',         # Vetting status
            'kepler_name',          # Kepler planet name (only for confirmed)
        }
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw KOI dataset from CSV."""
        logger.info(f"Loading raw data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        # Load with comment='#' to skip header comments
        df = pd.read_csv(self.data_path, comment='#')
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def extract_target_labels(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract and encode target labels from disposition column.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Encoded target labels (0=CANDIDATE, 1=CONFIRMED, 2=FALSE POSITIVE)
        """
        if 'koi_disposition' not in df.columns:
            raise ValueError("Target column 'koi_disposition' not found")
            
        # Map disposition to standard labels
        disposition_mapping = {
            'CANDIDATE': 'CANDIDATE',
            'CONFIRMED': 'CONFIRMED', 
            'FALSE POSITIVE': 'FALSE POSITIVE'
        }
        
        y = df['koi_disposition'].map(disposition_mapping)
        
        # Check for unmapped values
        unmapped = y.isnull().sum()
        if unmapped > 0:
            logger.warning(f"Found {unmapped} unmapped disposition values")
            y = y.dropna()
            
        # Encode labels: alphabetical order -> CANDIDATE=0, CONFIRMED=1, FALSE POSITIVE=2
        y_encoded = self.label_encoder.fit_transform(y)
        y_encoded_array = np.asarray(y_encoded)
        
        logger.info(f"Target label distribution:")
        for i, label in enumerate(self.label_encoder.classes_):
            count = int(np.sum(y_encoded_array == i))
            total = y_encoded_array.shape[0]
            logger.info(f"  {label}: {count} ({count/total*100:.1f}%)")
            
        return pd.Series(y_encoded_array, index=y.index, name='target')
    
    def extract_grouping_variable(self, df: pd.DataFrame) -> pd.Series:
        """
        Extract host star IDs for group-based splitting.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Series of Kepler IDs for grouping
        """
        if 'kepid' not in df.columns:
            raise ValueError("Grouping column 'kepid' not found")
            
        groups = df['kepid'].astype(str)
        groups.name = 'group'  # Standardize name for consistency
        logger.info(f"Found {groups.nunique()} unique host stars")
        
        return groups
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select and rename features, excluding label-leaking columns.
        
        Args:
            df: Raw dataframe
            
        Returns:
            DataFrame with selected features
        """
        # Get available features from mapping
        available_features = {}
        missing_features = []
        
        for feature_name, column_name in self.feature_mapping.items():
            if column_name in df.columns:
                available_features[feature_name] = column_name
            else:
                missing_features.append(f"{feature_name} ({column_name})")
                
        logger.info(f"Available features: {len(available_features)}/{len(self.feature_mapping)}")
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            
        # Select available columns and rename
        feature_df = df[list(available_features.values())].copy()
        feature_df.columns = list(available_features.keys())
        
        # Verify no label leakage
        leaked_columns = set(df.columns) & self.label_leak_columns
        if leaked_columns:
            logger.warning(f"Removing label-leaking columns: {leaked_columns}")
            
        return feature_df
    
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with imputation and missingness flags.
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame with imputed values and missingness indicators
        """
        X_processed = X.copy()
        
        # Create missingness indicator features
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(X) * 100
                logger.info(f"Column '{col}': {missing_count} missing ({missing_pct:.1f}%)")
                
                # Add missingness flag
                X_processed[f'{col}_missing'] = X[col].isnull().astype(int)
                
                # Impute based on data type and distribution
                if X[col].dtype in ['float64', 'int64']:
                    # Use median for numeric features
                    median_val = X[col].median()
                    X_processed[col] = X[col].fillna(median_val)
                else:
                    # Use mode for categorical features
                    mode_val = X[col].mode().iloc[0] if not X[col].mode().empty else 'unknown'
                    X_processed[col] = X[col].fillna(mode_val)
        
        # Convert specific categorical columns to numeric
        # quarters column contains binary string representation, convert to numeric
        if 'quarters' in X_processed.columns:
            # Convert quarters string to number of quarters (count of '1's)
            X_processed['quarters'] = X_processed['quarters'].apply(
                lambda x: str(x).count('1') if pd.notna(x) else 0
            ).astype(float)
                    
        return X_processed
    
    def normalize_skewed_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log10 transformation to highly skewed features.
        
        Args:
            X: Feature dataframe
            
        Returns:
            DataFrame with normalized features
        """
        X_normalized = X.copy()
        
        # Features that typically need log transformation
        log_features = [
            'period', 'depth', 'planet_radius', 'semi_major_axis', 
            'insolation', 'equilibrium_temp', 'stellar_mass'
        ]
        
        for feature in log_features:
            if feature in X_normalized.columns:
                # Only log-transform positive values
                positive_mask = X_normalized[feature] > 0
                if positive_mask.sum() > 0:
                    # Create log feature, fill NaNs with median of the log values
                    log_values = np.log10(X_normalized.loc[positive_mask, feature])
                    X_normalized[f'{feature}_log10'] = np.nan
                    X_normalized.loc[positive_mask, f'{feature}_log10'] = log_values
                    
                    # Fill remaining NaNs with median
                    median_log = np.median(log_values)
                    X_normalized[f'{feature}_log10'] = X_normalized[f'{feature}_log10'].fillna(median_log)
                    
                    logger.info(f"Created log10 feature: {feature}_log10")
                    
        return X_normalized
    
    def compute_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """
        Compute class weights to handle imbalance.
        
        Args:
            y: Target labels
            
        Returns:
            Dictionary of class weights
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        logger.info("Class weights for imbalance handling:")
        for class_idx, weight in class_weights.items():
            class_name = self.label_encoder.classes_[class_idx]
            logger.info(f"  {class_name}: {weight:.3f}")
            
        return class_weights
    
    def load_dataset(self, use_cache: bool = True, save_cache: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Load and preprocess the complete Kepler dataset.
        
        Args:
            use_cache: Whether to load from cached parquet if available
            save_cache: Whether to save processed data to cache
            
        Returns:
            Tuple of (X, y, groups, feature_names)
            - X: Feature matrix (DataFrame)
            - y: Target labels (Series) 
            - groups: Host star IDs for splitting (Series)
            - feature_names: List of feature column names
        """
        # Check cache first
        if use_cache and self.cache_path.exists():
            logger.info(f"Loading from cache: {self.cache_path}")
            try:
                cached_data = pd.read_parquet(self.cache_path)
                X = cached_data.drop(['target', 'group'], axis=1)
                y = cached_data['target'].rename('target')
                groups = cached_data['group'].rename('group')
                return X, y, groups, list(X.columns)
            except Exception as e:
                logger.warning(f"Cache loading failed: {e}, proceeding with fresh processing")
        
        # Load and process raw data
        logger.info("Processing raw dataset...")
        raw_df = self.load_raw_data()
        
        # Extract components
        y = self.extract_target_labels(raw_df)
        groups = self.extract_grouping_variable(raw_df)
        
        # Align indices (keep only rows with valid targets)
        valid_idx = y.index
        raw_df = raw_df.loc[valid_idx]
        groups = groups.loc[valid_idx]
        
        # Feature processing
        X = self.select_features(raw_df)
        X = self.handle_missing_values(X)
        X = self.normalize_skewed_features(X)
        
        # Final alignment
        X = X.loc[valid_idx]
        
        logger.info(f"Final dataset shape: {X.shape}")
        logger.info(f"Features: {list(X.columns)}")
        
        # Compute class weights for reference
        class_weights = self.compute_class_weights(y)
        
        # Cache processed data
        if save_cache:
            logger.info(f"Saving to cache: {self.cache_path}")
            self.cache_path.parent.mkdir(exist_ok=True)
            
            cached_df = X.copy()
            cached_df['target'] = y
            cached_df['group'] = groups
            cached_df.to_parquet(self.cache_path, index=False)
            
        return X, y, groups, list(X.columns)


def load_kepler_dataset(use_cache: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Convenience function to load Kepler dataset.
    
    Args:
        use_cache: Whether to use cached processed data
        
    Returns:
        Tuple of (X, y, groups, feature_names)
    """
    loader = KeplerDatasetLoader()
    return loader.load_dataset(use_cache=use_cache)


class K2DatasetLoader:
    """
    K2 Objects of Interest (K2OI) dataset loader and preprocessor.
    
    Handles data cleaning, feature engineering, and preprocessing for
    K2 exoplanet classification with proper label leakage prevention.
    """
    
    def __init__(self, data_path: str = "/Users/kkgogada/Code/NASASAC2025/K2OI.csv"):
        """
        Initialize the K2 dataset loader.
        
        Args:
            data_path: Path to the K2OI.csv file
        """
        self.data_path = Path(data_path)
        self.cache_path = Path("/Users/kkgogada/Code/NASASAC2025/dataset/k2_processed.parquet")
        self.label_encoder = LabelEncoder()
        
        # Define feature mapping for K2 dataset
        self.feature_mapping = {
            # Orbital characteristics
            'period': 'pl_orbper',
            'duration': 'pl_trandur',
            'depth': 'pl_trandep', 
            'impact_param': 'pl_imppar',
            
            # Planet characteristics
            'planet_radius': 'pl_rade',         # Planet radius in Earth radii
            'planet_mass': 'pl_masse',          # Planet mass in Earth masses
            'semi_major_axis': 'pl_orbsmax',    # Semi-major axis in AU
            'equilibrium_temp': 'pl_eqt',       # Equilibrium temperature
            'insolation': 'pl_insol',           # Insolation flux
            'eccentricity': 'pl_orbeccen',      # Orbital eccentricity
            'inclination': 'pl_orbincl',        # Orbital inclination
            
            # Transit properties
            'transit_duration': 'pl_trandur',   # Transit duration (hours)
            'transit_depth': 'pl_trandep',      # Transit depth (ppm)
            'radius_ratio': 'pl_ratror',        # Planet-to-star radius ratio
            
            # Stellar characteristics
            'stellar_teff': 'st_teff',          # Stellar effective temperature
            'stellar_logg': 'st_logg',          # Stellar surface gravity
            'stellar_metallicity': 'st_met',    # Stellar metallicity [Fe/H]
            'stellar_radius': 'st_rad',         # Stellar radius
            'stellar_mass': 'st_mass',          # Stellar mass
            'stellar_luminosity': 'st_lum',     # Stellar luminosity
            'stellar_age': 'st_age',            # Stellar age
            
            # Photometry
            'v_mag': 'sy_vmag',                 # V magnitude
            'j_mag': 'sy_jmag',                 # J magnitude  
            'h_mag': 'sy_hmag',                 # H magnitude
            'k_mag': 'sy_kmag',                 # K magnitude
            'kepler_mag': 'sy_kepmag',          # Kepler magnitude
            
            # System properties
            'distance': 'sy_dist',              # Distance to system (pc)
            'proper_motion': 'sy_pm',           # Proper motion
            'radial_velocity': 'st_radv',       # Radial velocity
        }
        
        # Columns that leak the target label
        self.label_leak_columns = {
            'disposition',          # Target disposition
            'disp_refname',        # Disposition reference
            'pl_controv_flag',     # Controversy flag
            'discoverymethod',     # Discovery method
            'disc_year',           # Discovery year
            'disc_refname',        # Discovery reference
            'pl_name',             # Planet name (only for confirmed)
        }
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw K2OI dataset from CSV."""
        logger.info(f"Loading raw K2 data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path, comment='#')
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def extract_target_labels(self, df: pd.DataFrame) -> pd.Series:
        """Extract and encode target labels from K2 disposition column."""
        if 'disposition' not in df.columns:
            raise ValueError("Target column 'disposition' not found")
        
        # Map K2 disposition to standard 3-class labels
        disposition_mapping = {
            'CANDIDATE': 'CANDIDATE',
            'CONFIRMED': 'CONFIRMED',
            'FALSE POSITIVE': 'FALSE POSITIVE',
            'REFUTED': 'FALSE POSITIVE',  # Map REFUTED to FALSE POSITIVE
        }
        
        y = df['disposition'].map(disposition_mapping)
        
        # Check for unmapped values
        unmapped = y.isnull().sum()
        if unmapped > 0:
            logger.warning(f"Found {unmapped} unmapped disposition values")
            # Keep only mapped values
            valid_mask = y.notna()
            y = y[valid_mask]
            df = df[valid_mask]
            
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_encoded_array = np.asarray(y_encoded)
        
        logger.info(f"K2 target label distribution:")
        for i, label in enumerate(self.label_encoder.classes_):
            count = int(np.sum(y_encoded_array == i))
            total = y_encoded_array.shape[0]
            logger.info(f"  {label}: {count} ({count/total*100:.1f}%)")
            
        return pd.Series(y_encoded_array, index=y.index, name='target')
    
    def extract_grouping_variable(self, df: pd.DataFrame) -> pd.Series:
        """Extract host star IDs for group-based splitting."""
        if 'epic_hostname' not in df.columns:
            raise ValueError("Grouping column 'epic_hostname' not found")
            
        groups = df['epic_hostname'].astype(str)
        groups.name = 'group'
        logger.info(f"Found {groups.nunique()} unique K2 host stars")
        
        return groups
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and rename features, excluding label-leaking columns."""
        available_features = {}
        missing_features = []
        
        for feature_name, column_name in self.feature_mapping.items():
            if column_name in df.columns:
                available_features[feature_name] = column_name
            else:
                missing_features.append(f"{feature_name} ({column_name})")
                
        logger.info(f"Available K2 features: {len(available_features)}/{len(self.feature_mapping)}")
        if missing_features:
            logger.warning(f"Missing K2 features: {missing_features[:5]}...")  # Show first 5
            
        # Select available columns and rename
        if available_features:
            feature_df = df[list(available_features.values())].copy()
            feature_df.columns = list(available_features.keys())
        else:
            logger.error("No features found in K2 dataset!")
            feature_df = pd.DataFrame(index=df.index)
            
        # Verify no label leakage
        leaked_columns = set(df.columns) & self.label_leak_columns
        if leaked_columns:
            logger.warning(f"Removing K2 label-leaking columns: {leaked_columns}")
            
        return feature_df
    
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with imputation and missingness flags."""
        X_processed = X.copy()
        
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(X) * 100
                logger.info(f"K2 Column '{col}': {missing_count} missing ({missing_pct:.1f}%)")
                
                # Add missingness flag
                X_processed[f'{col}_missing'] = X[col].isnull().astype(int)
                
                # Impute based on data type
                if X[col].dtype in ['float64', 'int64']:
                    median_val = X[col].median()
                    X_processed[col] = X[col].fillna(median_val)
                else:
                    mode_val = X[col].mode().iloc[0] if not X[col].mode().empty else 'unknown'
                    X_processed[col] = X[col].fillna(mode_val)
                    
        return X_processed
    
    def normalize_skewed_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log10 transformation to highly skewed features."""
        X_normalized = X.copy()
        
        log_features = [
            'period', 'depth', 'planet_radius', 'planet_mass', 
            'semi_major_axis', 'insolation', 'equilibrium_temp', 'distance'
        ]
        
        for feature in log_features:
            if feature in X_normalized.columns:
                positive_mask = X_normalized[feature] > 0
                if positive_mask.sum() > 0:
                    log_values = np.log10(X_normalized.loc[positive_mask, feature])
                    X_normalized[f'{feature}_log10'] = np.nan
                    X_normalized.loc[positive_mask, f'{feature}_log10'] = log_values
                    
                    # Fill remaining NaNs with median
                    median_log = np.median(log_values)
                    X_normalized[f'{feature}_log10'] = X_normalized[f'{feature}_log10'].fillna(median_log)
                    
                    logger.info(f"Created K2 log10 feature: {feature}_log10")
                    
        return X_normalized
    
    def compute_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Compute class weights to handle imbalance."""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        logger.info("K2 class weights for imbalance handling:")
        for class_idx, weight in class_weights.items():
            class_name = self.label_encoder.classes_[class_idx]
            logger.info(f"  {class_name}: {weight:.3f}")
            
        return class_weights
    
    def load_dataset(self, use_cache: bool = True, save_cache: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """Load and preprocess the complete K2 dataset."""
        # Check cache first
        if use_cache and self.cache_path.exists():
            logger.info(f"Loading K2 from cache: {self.cache_path}")
            try:
                cached_data = pd.read_parquet(self.cache_path)
                X = cached_data.drop(['target', 'group'], axis=1)
                y = cached_data['target'].rename('target')
                groups = cached_data['group'].rename('group')
                return X, y, groups, list(X.columns)
            except Exception as e:
                logger.warning(f"K2 cache loading failed: {e}, proceeding with fresh processing")
        
        # Load and process raw data
        logger.info("Processing raw K2 dataset...")
        raw_df = self.load_raw_data()
        
        # Extract components
        y = self.extract_target_labels(raw_df)
        groups = self.extract_grouping_variable(raw_df)
        
        # Align indices (keep only rows with valid targets)
        valid_idx = y.index
        raw_df = raw_df.loc[valid_idx]
        groups = groups.loc[valid_idx]
        
        # Feature processing
        X = self.select_features(raw_df)
        X = self.handle_missing_values(X)
        X = self.normalize_skewed_features(X)
        
        # Final alignment
        X = X.loc[valid_idx]
        
        logger.info(f"Final K2 dataset shape: {X.shape}")
        logger.info(f"K2 features: {len(X.columns)}")
        
        # Compute class weights
        class_weights = self.compute_class_weights(y)
        
        # Cache processed data
        if save_cache:
            logger.info(f"Saving K2 to cache: {self.cache_path}")
            self.cache_path.parent.mkdir(exist_ok=True)
            
            cached_df = X.copy()
            cached_df['target'] = y
            cached_df['group'] = groups
            cached_df.to_parquet(self.cache_path, index=False)
            
        return X, y, groups, list(X.columns)


class TESSDatasetLoader:
    """
    TESS Objects of Interest (TOI) dataset loader and preprocessor.
    
    Handles data cleaning, feature engineering, and preprocessing for
    TESS exoplanet classification with proper label leakage prevention.
    """
    
    def __init__(self, data_path: str = "/Users/kkgogada/Code/NASASAC2025/TESSOI.csv"):
        """
        Initialize the TESS dataset loader.
        
        Args:
            data_path: Path to the TESSOI.csv file
        """
        self.data_path = Path(data_path)
        self.cache_path = Path("/Users/kkgogada/Code/NASASAC2025/dataset/tess_processed.parquet")
        self.label_encoder = LabelEncoder()
        
        # Define feature mapping for TESS dataset
        self.feature_mapping = {
            # Orbital characteristics
            'period': 'pl_orbper',
            'duration': 'pl_trandurh',          # Transit duration (hours)
            'depth': 'pl_trandep',              # Transit depth (ppm)
            
            # Planet characteristics  
            'planet_radius': 'pl_rade',         # Planet radius in Earth radii
            'equilibrium_temp': 'pl_eqt',       # Equilibrium temperature
            'insolation': 'pl_insol',           # Insolation flux
            
            # Stellar characteristics
            'stellar_teff': 'st_teff',          # Stellar effective temperature
            'stellar_logg': 'st_logg',          # Stellar surface gravity
            'stellar_radius': 'st_rad',         # Stellar radius
            'stellar_mass': 'st_mass',          # Stellar mass
            'stellar_luminosity': 'st_lum',     # Stellar luminosity
            
            # TESS-specific
            'tess_mag': 'st_tmag',              # TESS magnitude
            'distance': 'st_dist',              # Distance to system
            
            # Coordinates and proper motion
            'ra': 'ra',                         # Right ascension
            'dec': 'dec',                       # Declination
            'proper_motion_ra': 'st_pmra',      # Proper motion in RA
            'proper_motion_dec': 'st_pmdec',    # Proper motion in Dec
        }
        
        # Columns that leak the target label
        self.label_leak_columns = {
            'tfopwg_disp',         # Target disposition
            'ctoi_alias',          # CTOI alias (relates to disposition)
            'pl_pnum',             # Pipeline signal ID (could leak info)
        }
    
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw TESS dataset from CSV."""
        logger.info(f"Loading raw TESS data from {self.data_path}")
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path, comment='#')
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def extract_target_labels(self, df: pd.DataFrame) -> pd.Series:
        """Extract and encode target labels from TESS disposition column."""
        if 'tfopwg_disp' not in df.columns:
            raise ValueError("Target column 'tfopwg_disp' not found")
        
        # Map TESS disposition codes to standard 3-class labels
        disposition_mapping = {
            'PC': 'CANDIDATE',          # Planet Candidate
            'APC': 'CANDIDATE',         # Ambiguous Planet Candidate
            'CP': 'CONFIRMED',          # Confirmed Planet
            'KP': 'CONFIRMED',          # Known Planet
            'FP': 'FALSE POSITIVE',     # False Positive
            'FA': 'FALSE POSITIVE',     # False Alarm
        }
        
        y = df['tfopwg_disp'].map(disposition_mapping)
        
        # Check for unmapped values
        unmapped = y.isnull().sum()
        if unmapped > 0:
            logger.warning(f"Found {unmapped} unmapped TESS disposition values")
            # Show unique unmapped values
            unmapped_values = df.loc[y.isnull(), 'tfopwg_disp'].unique()
            logger.warning(f"Unmapped values: {unmapped_values}")
            # Keep only mapped values
            valid_mask = y.notna()
            y = y[valid_mask]
            df = df[valid_mask]
            
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_encoded_array = np.asarray(y_encoded)
        
        logger.info(f"TESS target label distribution:")
        for i, label in enumerate(self.label_encoder.classes_):
            count = int(np.sum(y_encoded_array == i))
            total = y_encoded_array.shape[0]
            logger.info(f"  {label}: {count} ({count/total*100:.1f}%)")
            
        return pd.Series(y_encoded_array, index=y.index, name='target')
    
    def extract_grouping_variable(self, df: pd.DataFrame) -> pd.Series:
        """Extract host star IDs for group-based splitting."""
        if 'tid' not in df.columns:
            raise ValueError("Grouping column 'tid' not found")
            
        groups = df['tid'].astype(str)
        groups.name = 'group'
        logger.info(f"Found {groups.nunique()} unique TESS host stars")
        
        return groups
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and rename features, excluding label-leaking columns."""
        available_features = {}
        missing_features = []
        
        for feature_name, column_name in self.feature_mapping.items():
            if column_name in df.columns:
                available_features[feature_name] = column_name
            else:
                missing_features.append(f"{feature_name} ({column_name})")
                
        logger.info(f"Available TESS features: {len(available_features)}/{len(self.feature_mapping)}")
        if missing_features:
            logger.warning(f"Missing TESS features: {missing_features[:5]}...")  # Show first 5
            
        # Select available columns and rename
        if available_features:
            feature_df = df[list(available_features.values())].copy()
            feature_df.columns = list(available_features.keys())
        else:
            logger.error("No features found in TESS dataset!")
            feature_df = pd.DataFrame(index=df.index)
            
        # Verify no label leakage
        leaked_columns = set(df.columns) & self.label_leak_columns
        if leaked_columns:
            logger.warning(f"Removing TESS label-leaking columns: {leaked_columns}")
            
        return feature_df
    
    def handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with imputation and missingness flags."""
        X_processed = X.copy()
        
        for col in X.columns:
            missing_count = X[col].isnull().sum()
            if missing_count > 0:
                missing_pct = missing_count / len(X) * 100
                logger.info(f"TESS Column '{col}': {missing_count} missing ({missing_pct:.1f}%)")
                
                # Add missingness flag
                X_processed[f'{col}_missing'] = X[col].isnull().astype(int)
                
                # Impute based on data type
                if X[col].dtype in ['float64', 'int64']:
                    median_val = X[col].median()
                    X_processed[col] = X[col].fillna(median_val)
                else:
                    mode_val = X[col].mode().iloc[0] if not X[col].mode().empty else 'unknown'
                    X_processed[col] = X[col].fillna(mode_val)
                    
        return X_processed
    
    def normalize_skewed_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log10 transformation to highly skewed features."""
        X_normalized = X.copy()
        
        log_features = [
            'period', 'depth', 'planet_radius', 'insolation', 
            'equilibrium_temp', 'distance', 'stellar_luminosity'
        ]
        
        for feature in log_features:
            if feature in X_normalized.columns:
                positive_mask = X_normalized[feature] > 0
                if positive_mask.sum() > 0:
                    log_values = np.log10(X_normalized.loc[positive_mask, feature])
                    X_normalized[f'{feature}_log10'] = np.nan
                    X_normalized.loc[positive_mask, f'{feature}_log10'] = log_values
                    
                    # Fill remaining NaNs with median
                    median_log = np.median(log_values)
                    X_normalized[f'{feature}_log10'] = X_normalized[f'{feature}_log10'].fillna(median_log)
                    
                    logger.info(f"Created TESS log10 feature: {feature}_log10")
                    
        return X_normalized
    
    def compute_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Compute class weights to handle imbalance."""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = dict(zip(classes, weights))
        
        logger.info("TESS class weights for imbalance handling:")
        for class_idx, weight in class_weights.items():
            class_name = self.label_encoder.classes_[class_idx]
            logger.info(f"  {class_name}: {weight:.3f}")
            
        return class_weights
    
    def load_dataset(self, use_cache: bool = True, save_cache: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """Load and preprocess the complete TESS dataset."""
        # Check cache first
        if use_cache and self.cache_path.exists():
            logger.info(f"Loading TESS from cache: {self.cache_path}")
            try:
                cached_data = pd.read_parquet(self.cache_path)
                X = cached_data.drop(['target', 'group'], axis=1)
                y = cached_data['target'].rename('target')
                groups = cached_data['group'].rename('group')
                return X, y, groups, list(X.columns)
            except Exception as e:
                logger.warning(f"TESS cache loading failed: {e}, proceeding with fresh processing")
        
        # Load and process raw data
        logger.info("Processing raw TESS dataset...")
        raw_df = self.load_raw_data()
        
        # Extract components
        y = self.extract_target_labels(raw_df)
        groups = self.extract_grouping_variable(raw_df)
        
        # Align indices (keep only rows with valid targets)
        valid_idx = y.index
        raw_df = raw_df.loc[valid_idx]
        groups = groups.loc[valid_idx]
        
        # Feature processing
        X = self.select_features(raw_df)
        X = self.handle_missing_values(X)
        X = self.normalize_skewed_features(X)
        
        # Final alignment
        X = X.loc[valid_idx]
        
        logger.info(f"Final TESS dataset shape: {X.shape}")
        logger.info(f"TESS features: {len(X.columns)}")
        
        # Compute class weights
        class_weights = self.compute_class_weights(y)
        
        # Cache processed data
        if save_cache:
            logger.info(f"Saving TESS to cache: {self.cache_path}")
            self.cache_path.parent.mkdir(exist_ok=True)
            
            cached_df = X.copy()
            cached_df['target'] = y
            cached_df['group'] = groups
            cached_df.to_parquet(self.cache_path, index=False)
            
        return X, y, groups, list(X.columns)


def load_k2_dataset(use_cache: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Convenience function to load K2 dataset."""
    loader = K2DatasetLoader()
    return loader.load_dataset(use_cache=use_cache)


def load_tess_dataset(use_cache: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """Convenience function to load TESS dataset."""
    loader = TESSDatasetLoader()
    return loader.load_dataset(use_cache=use_cache)


def load_all_datasets(use_cache: bool = True) -> Dict[str, Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]]:
    """Load all three NASA exoplanet datasets."""
    logger.info("Loading all NASA exoplanet datasets...")
    
    datasets = {}
    
    # Load Kepler
    try:
        datasets['kepler'] = load_kepler_dataset(use_cache=use_cache)
        logger.info(f"✅ Kepler dataset loaded: {datasets['kepler'][0].shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load Kepler dataset: {e}")
    
    # Load K2
    try:
        datasets['k2'] = load_k2_dataset(use_cache=use_cache)
        logger.info(f"✅ K2 dataset loaded: {datasets['k2'][0].shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load K2 dataset: {e}")
    
    # Load TESS
    try:
        datasets['tess'] = load_tess_dataset(use_cache=use_cache)
        logger.info(f"✅ TESS dataset loaded: {datasets['tess'][0].shape}")
    except Exception as e:
        logger.error(f"❌ Failed to load TESS dataset: {e}")
    
    logger.info(f"Loaded {len(datasets)}/3 datasets successfully")
    return datasets


if __name__ == "__main__":
    # Example usage - load all datasets
    datasets = load_all_datasets(use_cache=False)
    
    print(f"\n🚀 All NASA Exoplanet Datasets Loaded!")
    for mission, (X, y, groups, features) in datasets.items():
        print(f"\n{mission.upper()}:")
        print(f"  Shape: {X.shape}")
        print(f"  Features: {len(features)}")
        print(f"  Target distribution: {pd.Series(y).value_counts().sort_index()}")
        print(f"  Unique groups: {groups.nunique()}")