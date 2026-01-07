#!/usr/bin/env python3
"""
Script 22: Merge Structure + Bio Features (Step 4)

Merges ECFP matrices with biological features, applies proper scaling/imputation
following strict leakage prevention (fit only on training data).

Usage:
    python scripts/22_merge_features.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import json
from datetime import datetime
import hashlib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureMerger:
    def __init__(self):
        self.processing_log = {
            'timestamp': datetime.now().isoformat(),
            'scalers': {},
            'imputers': {},
            'feature_counts': {},
            'merge_stats': {}
        }
        
        self.continuous_scaler = None
        self.continuous_imputer = None
        self.binary_imputer = None
        
    def load_structure_features(self, split_name):
        """Load ECFP structure features."""
        struct_path = Path(f"data/work/X_struct_{split_name}.npy")
        
        if not struct_path.exists():
            logger.warning(f"Structure features not found: {struct_path}")
            return None
            
        X_struct = np.load(struct_path)
        logger.info(f"Loaded {split_name} structure features: {X_struct.shape}")
        return X_struct
    
    def load_bio_features(self, split_name):
        """Load biological features."""
        bio_path = Path(f"data/work/bio_{split_name}.parquet")
        
        if not bio_path.exists():
            logger.warning(f"Bio features not found: {bio_path}")
            return None
            
        bio_df = pd.read_parquet(bio_path)
        logger.info(f"Loaded {split_name} bio features: {bio_df.shape}")
        return bio_df
    
    def identify_feature_types(self, bio_df):
        """Identify continuous vs binary features."""
        continuous_features = []
        binary_features = []
        missing_flags = []
        id_columns = []
        
        for col in bio_df.columns:
            if col in ['InChIKey', 'SMILES', 'compound_id']:
                id_columns.append(col)
            elif col.endswith('_is_missing'):
                missing_flags.append(col)
            elif bio_df[col].dtype == 'object':
                continue  # Skip non-numeric
            elif set(bio_df[col].dropna().unique()).issubset({0, 1, 0.0, 1.0}):
                binary_features.append(col)
            else:
                continuous_features.append(col)
        
        logger.info(f"Feature types identified:")
        logger.info(f"  Continuous: {len(continuous_features)}")
        logger.info(f"  Binary: {len(binary_features)}")
        logger.info(f"  Missing flags: {len(missing_flags)}")
        logger.info(f"  ID columns: {len(id_columns)}")
        
        return continuous_features, binary_features, missing_flags, id_columns
    
    def fit_preprocessors(self, bio_df_train):
        """Fit imputers and scalers on training data only."""
        logger.info("Fitting preprocessors on training data...")
        
        continuous_features, binary_features, missing_flags, _ = self.identify_feature_types(bio_df_train)
        
        # Fit continuous feature preprocessors
        if continuous_features:
            # Imputer for continuous features
            self.continuous_imputer = SimpleImputer(strategy='median')
            continuous_data = bio_df_train[continuous_features].values
            self.continuous_imputer.fit(continuous_data)
            
            # Scaler for continuous features (after imputation)
            imputed_continuous = self.continuous_imputer.transform(continuous_data)
            self.continuous_scaler = StandardScaler()
            self.continuous_scaler.fit(imputed_continuous)
            
            logger.info(f"Fitted continuous imputer and scaler for {len(continuous_features)} features")
        
        # Fit binary feature imputer
        if binary_features:
            self.binary_imputer = SimpleImputer(strategy='most_frequent')
            binary_data = bio_df_train[binary_features].values
            self.binary_imputer.fit(binary_data)
            
            logger.info(f"Fitted binary imputer for {len(binary_features)} features")
        
        # Store feature lists for later use
        self.continuous_features = continuous_features
        self.binary_features = binary_features
        self.missing_flags = missing_flags
        
        # Log preprocessing info
        self.processing_log['scalers']['continuous'] = {
            'type': 'StandardScaler',
            'n_features': len(continuous_features),
            'feature_names': continuous_features
        }
        self.processing_log['imputers']['continuous'] = {
            'strategy': 'median',
            'n_features': len(continuous_features)
        }
        self.processing_log['imputers']['binary'] = {
            'strategy': 'most_frequent',
            'n_features': len(binary_features)
        }
    
    def transform_bio_features(self, bio_df, split_name):
        """Transform bio features using fitted preprocessors."""
        logger.info(f"Transforming bio features for {split_name}...")
        
        # Start with missing flags (no transformation needed)
        transformed_features = []
        feature_names = []
        
        # Add missing flags
        if self.missing_flags:
            missing_data = bio_df[self.missing_flags].values.astype(np.float32)
            transformed_features.append(missing_data)
            feature_names.extend(self.missing_flags)
        
        # Transform continuous features
        if self.continuous_features and self.continuous_imputer is not None:
            continuous_data = bio_df[self.continuous_features].values
            imputed_continuous = self.continuous_imputer.transform(continuous_data)
            scaled_continuous = self.continuous_scaler.transform(imputed_continuous)
            transformed_features.append(scaled_continuous.astype(np.float32))
            feature_names.extend(self.continuous_features)
        
        # Transform binary features
        if self.binary_features and self.binary_imputer is not None:
            binary_data = bio_df[self.binary_features].values
            imputed_binary = self.binary_imputer.transform(binary_data)
            transformed_features.append(imputed_binary.astype(np.float32))
            feature_names.extend(self.binary_features)
        
        # Concatenate all features
        if transformed_features:
            X_bio = np.concatenate(transformed_features, axis=1)
        else:
            X_bio = np.empty((len(bio_df), 0), dtype=np.float32)
        
        logger.info(f"Transformed bio features shape: {X_bio.shape}")
        
        return X_bio, feature_names
    
    def merge_features(self, X_struct, X_bio, split_name):
        """Merge structure and bio features."""
        if X_struct is None or X_bio is None:
            logger.error(f"Cannot merge features for {split_name}: missing data")
            return None, None
        
        # Check sample alignment
        if X_struct.shape[0] != X_bio.shape[0]:
            logger.error(f"Sample count mismatch for {split_name}: struct={X_struct.shape[0]}, bio={X_bio.shape[0]}")
            return None, None
        
        # Concatenate features
        X_mixed = np.concatenate([X_struct, X_bio], axis=1)
        
        # Create feature names
        n_struct_features = X_struct.shape[1]
        n_bio_features = X_bio.shape[1]
        
        struct_names = [f"ecfp_{i}" for i in range(n_struct_features)]
        bio_names = getattr(self, 'current_bio_names', [f"bio_{i}" for i in range(n_bio_features)])
        
        feature_names = struct_names + bio_names
        
        logger.info(f"Merged features for {split_name}: {X_mixed.shape}")
        logger.info(f"  Structure: {n_struct_features} features")
        logger.info(f"  Bio: {n_bio_features} features")
        
        # Update processing log
        self.processing_log['merge_stats'][split_name] = {
            'n_samples': X_mixed.shape[0],
            'n_struct_features': n_struct_features,
            'n_bio_features': n_bio_features,
            'n_total_features': X_mixed.shape[1]
        }
        
        return X_mixed, feature_names
    
    def save_merged_features(self, X_mixed, feature_names, split_name):
        """Save merged features to disk."""
        if X_mixed is None:
            return
            
        # Save feature matrix
        output_path = Path(f"data/work/X_mixed_{split_name}.npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        np.savez_compressed(output_path, X=X_mixed)
        logger.info(f"Saved mixed features to {output_path}")
        
        # Save feature names (only once for consistency)
        if split_name == 'train':
            names_path = Path("data/work/feature_names_mixed.json")
            with open(names_path, 'w') as f:
                json.dump(feature_names, f, indent=2)
            logger.info(f"Saved feature names to {names_path}")
    
    def save_preprocessors(self):
        """Save fitted preprocessors for later use."""
        preprocessor_path = Path("data/work/bio_preprocessors.pkl")
        
        preprocessors = {
            'continuous_imputer': self.continuous_imputer,
            'continuous_scaler': self.continuous_scaler,
            'binary_imputer': self.binary_imputer,
            'continuous_features': self.continuous_features,
            'binary_features': self.binary_features,
            'missing_flags': self.missing_flags
        }
        
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(preprocessors, f)
        
        logger.info(f"Saved preprocessors to {preprocessor_path}")

def main():
    """Main merging pipeline."""
    logger.info("Starting structure + bio feature merging...")
    
    # Initialize merger
    merger = FeatureMerger()
    
    # Define splits
    splits = ['train', 'dev', 'test']
    
    # First pass: load training data and fit preprocessors
    logger.info("=== FITTING PREPROCESSORS (Training data only) ===")
    bio_df_train = merger.load_bio_features('train')
    
    if bio_df_train is not None:
        merger.fit_preprocessors(bio_df_train)
        merger.save_preprocessors()
    else:
        logger.error("Cannot proceed without training bio features")
        return
    
    # Second pass: transform and merge all splits
    logger.info("=== TRANSFORMING AND MERGING ALL SPLITS ===")
    
    for split_name in splits:
        logger.info(f"\nProcessing {split_name} split...")
        
        # Load features
        X_struct = merger.load_structure_features(split_name)
        bio_df = merger.load_bio_features(split_name)
        
        if bio_df is None:
            logger.warning(f"Skipping {split_name}: no bio features")
            continue
        
        # Transform bio features
        X_bio, bio_feature_names = merger.transform_bio_features(bio_df, split_name)
        merger.current_bio_names = bio_feature_names  # Store for merging
        
        # Merge features
        X_mixed, feature_names = merger.merge_features(X_struct, X_bio, split_name)
        
        # Save merged features
        merger.save_merged_features(X_mixed, feature_names, split_name)
    
    # Save processing log
    log_path = Path("reports/logs/feature_merging.json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, 'w') as f:
        json.dump(merger.processing_log, f, indent=2)
    
    logger.info(f"Saved processing log to {log_path}")
    logger.info("Feature merging completed successfully!")

if __name__ == "__main__":
    main()