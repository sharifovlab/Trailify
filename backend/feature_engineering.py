"""
Shared Feature Engineering Module
This ensures train/serve consistency by using the SAME feature engineering logic
in both MODEL.py (training) and app.py (serving)
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def add_derived_features(df):
    """
    Apply feature engineering transformations
    
    This function is used in BOTH:
    1. Training (MODEL.py) - to create features from raw trail data
    2. Serving (app.py) - to create features from GPX analysis
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with base features
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added derived features
    """
    df = df.copy()
    
    try:
        # Distance in kilometers (with safety check)
        df['distance_km'] = np.maximum(df['total_distance_m'] / 1000, 0.001)
        
        # Elevation per kilometer (safe division)
        df['elevation_per_km'] = np.divide(
            df['total_elevation_gain_m'],
            df['distance_km'],
            out=np.zeros_like(df['distance_km'], dtype=float),
            where=df['distance_km'] > 0.001
        )
        
        # Average grade
        df['avg_grade'] = (df['max_grade_pct'] + np.abs(df['min_grade_pct'])) / 2
        
        # Grade range
        df['grade_range'] = df['max_grade_pct'] - df['min_grade_pct']
        
        # Steep ratio (with safety for division)
        denominator = df['pct_segments_steep_gt_10'] + 1
        df['steep_ratio'] = df['pct_segments_steep_gt_15'] / denominator
        
        # Technical index
        df['technical_index'] = df['pct_sharp_turns'] * df['turn_max_deg'] / 100
        
        # Surface difficulty weighted score
        df['surface_difficulty'] = (
            df['surface_unpaved_pct'] * 0.5 + 
            df['surface_dirt_pct'] * 0.3 + 
            df['surface_ground_pct'] * 0.2
        )
        
        logger.info(f"Successfully created {7} derived features")
        
        # Validate no NaN or Inf values were created
        derived_cols = [
            'distance_km', 'elevation_per_km', 'avg_grade', 
            'grade_range', 'steep_ratio', 'technical_index', 'surface_difficulty'
        ]
        
        for col in derived_cols:
            if df[col].isnull().any():
                logger.warning(f"NaN values detected in {col}")
            if np.isinf(df[col]).any():
                logger.warning(f"Inf values detected in {col}")
                # Replace inf with large number
                df[col] = df[col].replace([np.inf, -np.inf], 0)
        
        return df
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise ValueError(f"Feature engineering error: {str(e)}")


def get_base_feature_columns():
    """
    Returns list of base features expected from GPX analysis or training data
    
    These are the raw features BEFORE derived features are added
    """
    return [
        'total_distance_m',
        'sinuosity',
        'elevation_range_m',
        'total_elevation_gain_m',
        'total_elevation_loss_m',
        'climb_rate_m_per_km',
        'descent_rate_m_per_km',
        'max_grade_pct',
        'min_grade_pct',
        'pct_segments_steep_gt_10',
        'pct_segments_steep_gt_15',
        'pct_steep_uphill',
        'pct_steep_downhill',
        'turn_count',
        'turn_mean_deg',
        'turn_std_deg',
        'turn_max_deg',
        'num_sharp_turns',
        'pct_sharp_turns',
        'turns_deg_per_km',
        'surface_unpaved_pct',
        'surface_dirt_pct',
        'surface_ground_pct'
    ]


def get_all_feature_columns():
    """
    Returns list of ALL features (base + derived) used by the model
    """
    base_features = get_base_feature_columns()
    derived_features = [
        'distance_km',
        'elevation_per_km',
        'avg_grade',
        'grade_range',
        'steep_ratio',
        'technical_index',
        'surface_difficulty'
    ]
    return base_features + derived_features


def validate_features(df, required_features=None):
    """
    Validates that all required features are present and valid
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to validate
    required_features : list, optional
        List of required feature names. If None, uses all features.
    
    Returns:
    --------
    tuple: (is_valid: bool, missing_features: list, invalid_features: dict)
    """
    if required_features is None:
        required_features = get_all_feature_columns()
    
    # Check for missing columns
    missing_features = [f for f in required_features if f not in df.columns]
    
    # Check for invalid values (NaN, Inf)
    invalid_features = {}
    for col in required_features:
        if col in df.columns:
            if df[col].isnull().any():
                invalid_features[col] = 'contains NaN'
            elif np.isinf(df[col]).any():
                invalid_features[col] = 'contains Inf'
    
    is_valid = len(missing_features) == 0 and len(invalid_features) == 0
    
    return is_valid, missing_features, invalid_features


def get_surface_features_from_type(surface_type):
    """
    Maps surface type to feature percentages
    
    Parameters:
    -----------
    surface_type : str
        One of: 'paved', 'gravel', 'dirt', 'technical', 'mixed'
    
    Returns:
    --------
    dict: Surface feature values
    """
    surface_mappings = {
        'paved': {
            'surface_unpaved_pct': 0.0,
            'surface_dirt_pct': 0.0,
            'surface_ground_pct': 0.0
        },
        'gravel': {
            'surface_unpaved_pct': 80.0,
            'surface_dirt_pct': 15.0,
            'surface_ground_pct': 5.0
        },
        'dirt': {
            'surface_unpaved_pct': 30.0,
            'surface_dirt_pct': 60.0,
            'surface_ground_pct': 10.0
        },
        'technical': {
            'surface_unpaved_pct': 40.0,
            'surface_dirt_pct': 30.0,
            'surface_ground_pct': 30.0
        },
        'mixed': {
            'surface_unpaved_pct': 50.0,
            'surface_dirt_pct': 30.0,
            'surface_ground_pct': 20.0
        }
    }
    
    surface_type_lower = surface_type.lower() if surface_type else 'mixed'
    
    if surface_type_lower not in surface_mappings:
        logger.warning(f"Unknown surface type '{surface_type}', using 'mixed'")
        surface_type_lower = 'mixed'
    
    return surface_mappings[surface_type_lower]