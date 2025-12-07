import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from datetime import datetime
import os
import json

# Import shared feature engineering
from backend.feature_engineering import add_derived_features, get_base_feature_columns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrailDifficultyModel:
    """
    Production-ready XGBoost model for trail difficulty prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_importance = None
        self.training_metadata = {}
        
    def prepare_features(self, df):
        """
        Prepares features for training using shared feature engineering
        """
        # Get base features
        base_features = get_base_feature_columns()
        
        # Validate base features exist
        missing = [f for f in base_features if f not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Add derived features using shared module
        df_features = add_derived_features(df[base_features])
        
        # Get all feature columns (base + derived)
        self.feature_columns = list(df_features.columns)
        
        logger.info(f"Prepared {len(self.feature_columns)} features")
        return df_features
    
    def train(self, excel_file, target_column='difficulty_score', test_size=0.2, 
              remove_low_importance=False, importance_threshold=0.01):
        """
        Trains XGBoost model with production best practices
        
        Parameters:
        -----------
        excel_file : str
            Path to Excel file with training data
        target_column : str
            Target variable column name
        test_size : float
            Test set size (0-1)
        remove_low_importance : bool
            Whether to remove low importance features and retrain
        importance_threshold : float
            Minimum feature importance to keep (if remove_low_importance=True)
        """
        logger.info("="*60)
        logger.info("STARTING MODEL TRAINING")
        logger.info("="*60)
        
        # Load data
        logger.info(f"Loading data from {excel_file}...")
        try:
            df = pd.read_excel(excel_file)
            logger.info(f"✓ Loaded {len(df)} trails")
        except Exception as e:
            logger.error(f"Failed to load data: {str(e)}")
            raise
        
        # Validate target column
        if target_column not in df.columns:
            logger.error(f"Target column '{target_column}' not found")
            logger.info("Available columns: " + ", ".join(df.columns.tolist()))
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        # Prepare features
        logger.info("\nPreparing features...")
        X = self.prepare_features(df)
        y = df[target_column]
        
        # Check for missing values
        if X.isnull().any().any():
            logger.warning("Missing values detected in features")
            missing_cols = X.columns[X.isnull().any()].tolist()
            logger.warning(f"Columns with missing values: {missing_cols}")
            logger.info("Filling missing values with median...")
            X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        logger.info(f"\nDataset split:")
        logger.info(f"  Training: {len(X_train)} samples")
        logger.info(f"  Testing:  {len(X_test)} samples")
        logger.info(f"  Features: {len(self.feature_columns)}")
        
        # Check if we have enough data
        if len(X_train) < 100:
            logger.warning(f"⚠️  Small training set: {len(X_train)} samples")
            logger.warning("Consider collecting more data for better generalization")
        
        # Normalize features
        logger.info("\nNormalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logger.info("\nTraining XGBoost model...")
        self.model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective='reg:squarederror',
            n_jobs=-1
        )
        
        # Train with validation set
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        
        # Log results
        logger.info("\n" + "="*60)
        logger.info("TRAINING RESULTS")
        logger.info("="*60)
        logger.info("\nTraining Set:")
        logger.info(f"  R² Score:  {train_r2:.4f}")
        logger.info(f"  RMSE:      {train_rmse:.4f}")
        logger.info(f"  MAE:       {train_mae:.4f}")
        
        logger.info("\nTest Set:")
        logger.info(f"  R² Score:  {test_r2:.4f}")
        logger.info(f"  RMSE:      {test_rmse:.4f}")
        logger.info(f"  MAE:       {test_mae:.4f}")
        
        # Check for overfitting
        r2_gap = train_r2 - test_r2
        if r2_gap > 0.15:
            logger.warning(f"\n⚠️  Potential overfitting detected!")
            logger.warning(f"   Train R² - Test R² = {r2_gap:.4f}")
            logger.warning("   Consider: collecting more data, reducing features, or tuning regularization")
        
        # Cross-validation
        logger.info("\nPerforming cross-validation...")
        cv_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            objective='reg:squarederror',
            n_jobs=-1
        )
        
        cv_scores = cross_val_score(
            cv_model, X_train_scaled, y_train, 
            cv=5, scoring='r2', n_jobs=-1
        )
        
        logger.info(f"\n5-Fold Cross-Validation R²:")
        logger.info(f"  Mean:  {cv_scores.mean():.4f}")
        logger.info(f"  Std:   {cv_scores.std():.4f}")
        logger.info(f"  Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\n" + "="*60)
        logger.info("TOP 15 IMPORTANT FEATURES")
        logger.info("="*60)
        for idx, row in self.feature_importance.head(15).iterrows():
            logger.info(f"  {row['feature']:30s} {row['importance']:.4f}")
        
        # Check for low importance features
        low_importance = self.feature_importance[
            self.feature_importance['importance'] < importance_threshold
        ]
        
        if len(low_importance) > 0:
            logger.info(f"\n⚠️  {len(low_importance)} features have importance < {importance_threshold}")
            logger.info("Low importance features:")
            for _, row in low_importance.iterrows():
                logger.info(f"  - {row['feature']}: {row['importance']:.4f}")
            
            if remove_low_importance:
                logger.info("\nRetraining with important features only...")
                return self._retrain_with_important_features(
                    X, y, test_size, importance_threshold
                )
        
        # Store training metadata
        self.training_metadata = {
            'training_date': datetime.now().isoformat(),
            'num_samples': len(df),
            'num_features': len(self.feature_columns),
            'test_size': test_size,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_mean_r2': cv_scores.mean(),
            'cv_std_r2': cv_scores.std(),
            'target_column': target_column
        }
        
        return {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'cv_mean_r2': cv_scores.mean(),
            'cv_std_r2': cv_scores.std()
        }
    
    def _retrain_with_important_features(self, X, y, test_size, threshold):
        """Retrain model using only important features"""
        important_features = self.feature_importance[
            self.feature_importance['importance'] >= threshold
        ]['feature'].tolist()
        
        logger.info(f"Retraining with {len(important_features)} important features...")
        
        # Update feature columns
        self.feature_columns = important_features
        X_important = X[important_features]
        
        # Retrain
        X_train, X_test, y_train, y_test = train_test_split(
            X_important, y, test_size=test_size, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Metrics
        y_pred_test = self.model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        logger.info(f"\nRetrained model metrics:")
        logger.info(f"  Test R²:   {test_r2:.4f}")
        logger.info(f"  Test RMSE: {test_rmse:.4f}")
        
        return {'test_r2': test_r2, 'test_rmse': test_rmse}
    
    def predict(self, df):
        """Predict difficulty for new trails"""
        if self.model is None:
            raise ValueError("Model not trained! Call train() first.")
        
        X = self.prepare_features(df)
        X = X[self.feature_columns]  # Ensure correct feature order
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return predictions
    
    def save_model(self, filepath=None, version=None):
        """
        Save model with versioning
        
        Parameters:
        -----------
        filepath : str, optional
            Custom filepath. If None, auto-generates versioned filename
        version : str, optional
            Version string (e.g., 'v1.2.0'). If None, uses timestamp
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        # Generate filename with version
        if filepath is None:
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            os.makedirs('models', exist_ok=True)
            filepath = f'models/trail_model_{version}.pkl'
        
        # Prepare model data
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'metadata': self.training_metadata
        }
        
        # Save
        joblib.dump(model_data, filepath)
        logger.info(f"\n💾 Model saved: {filepath}")
        
        # Save metadata as JSON for easy inspection
        metadata_path = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(self.training_metadata, f, indent=2)
        logger.info(f"📋 Metadata saved: {metadata_path}")
        
        # Save feature importance as CSV
        importance_path = filepath.replace('.pkl', '_feature_importance.csv')
        self.feature_importance.to_csv(importance_path, index=False)
        logger.info(f"📊 Feature importance saved: {importance_path}")
        
        return filepath
    
    def load_model(self, filepath):
        """Load trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            self.feature_importance = model_data.get('feature_importance')
            self.training_metadata = model_data.get('metadata', {})
            
            logger.info(f"\n📂 Model loaded: {filepath}")
            logger.info(f"   Features: {len(self.feature_columns)}")
            if self.training_metadata:
                logger.info(f"   Trained: {self.training_metadata.get('training_date', 'Unknown')}")
                logger.info(f"   Test R²: {self.training_metadata.get('test_r2', 'N/A')}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def plot_feature_importance(self, top_n=15, save_path=None):
        """Visualize feature importance"""
        if self.feature_importance is None:
            raise ValueError("No feature importance data available")
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        plt.title('Feature Importance (XGBoost)', fontsize=16, weight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Feature importance plot saved: {save_path}")
        
        plt.show()


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Trail Difficulty Model')
    parser.add_argument('--data', type=str, default='trails_with_difficulty.xlsx',
                       help='Path to training data (Excel file)')
    parser.add_argument('--target', type=str, default='difficulty_score',
                       help='Target column name')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0-1)')
    parser.add_argument('--version', type=str, default=None,
                       help='Model version (e.g., v1.0.0)')
    parser.add_argument('--remove-low-importance', action='store_true',
                       help='Remove low importance features and retrain')
    parser.add_argument('--importance-threshold', type=float, default=0.01,
                       help='Feature importance threshold')
    
    args = parser.parse_args()
    
    # Initialize model
    model = TrailDifficultyModel()
    
    # Train
    logger.info("="*60)
    logger.info("TRAIL DIFFICULTY MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Data file: {args.data}")
    logger.info(f"Target: {args.target}")
    logger.info(f"Test size: {args.test_size}")
    logger.info("="*60)
    
    try:
        results = model.train(
            args.data,
            target_column=args.target,
            test_size=args.test_size,
            remove_low_importance=args.remove_low_importance,
            importance_threshold=args.importance_threshold
        )
        
        # Save model
        model_path = model.save_model(version=args.version)
        
        # Plot feature importance
        try:
            importance_plot_path = model_path.replace('.pkl', '_importance.png')
            model.plot_feature_importance(save_path=importance_plot_path)
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {str(e)}")
        
        logger.info("\n" + "="*60)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"\nModel saved to: {model_path}")
        logger.info("\nTo use this model in production:")
        logger.info(f"  export MODEL_PATH='{model_path}'")
        logger.info("  python app.py")
        
    except Exception as e:
        logger.error(f"\n❌ TRAINING FAILED: {str(e)}", exc_info=True)
        raise