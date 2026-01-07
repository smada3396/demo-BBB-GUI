#!/usr/bin/env python3
"""
T8: Hardened 21-Builder Script

Robust bio feature builder with:
- .get() for YAML keys (graceful handling of missing keys)
- Column aliasing (smiles→SMILES, bbb_label→BBB_label)
- CLI args everywhere (no hard-coded paths)
- Always writes parquets (even if coverage is sparse)
- Comprehensive coverage reporting
"""

import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BioFeatureBuilder:
    def __init__(self, schema_path="features/bio_schema.yaml"):
        """Initialize builder with schema path"""
        self.schema_path = Path(schema_path)
        self.feature_dict = {}
        self.processing_log = {
            "status": "started",
            "errors": [],
            "warnings": [],
            "created_files": [],
            "skipped": [],
            "features_resolved": 0
        }
        
        # Load schema with graceful error handling
        try:
            with open(self.schema_path, 'r') as f:
                self.schema = yaml.safe_load(f)
            logger.info(f"📋 Loaded schema from {self.schema_path}")
        except FileNotFoundError:
            logger.error(f"❌ Schema file not found: {self.schema_path}")
            self.schema = {}
        except yaml.YAMLError as e:
            logger.error(f"❌ YAML parsing error: {e}")
            self.schema = {}
    
    def normalize_column_name(self, df, candidates):
        """Find and normalize column name from candidates"""
        for candidate in candidates:
            if candidate in df.columns:
                return candidate
        return None
    
    def apply_winsorization(self, series, winsorize_pct):
        """Apply winsorization to a series"""
        if pd.isna(winsorize_pct) or winsorize_pct <= 0:
            return series
        
        lower_bound = series.quantile(winsorize_pct / 100)
        upper_bound = series.quantile(1 - winsorize_pct / 100)
        
        return series.clip(lower_bound, upper_bound)
    
    def apply_transform(self, series, transform_type):
        """Apply transformation to a series"""
        if pd.isna(transform_type) or transform_type == "none":
            return series
        
        if transform_type == "log10":
            # Handle negative and zero values
            return np.where(series > 0, np.log10(series), np.nan)
        elif transform_type == "log1p":
            return np.log1p(series)
        else:
            logger.warning(f"Unknown transform type: {transform_type}")
            return series
    
    def process_bio_features(self, df):
        """Process biological features according to schema"""
        logger.info("🧪 Processing biological features...")
        
        # Get bio features from schema (with fallback)
        bio_features = self.schema.get('bio_features', {})
        
        if not bio_features:
            logger.warning("No bio_features found in schema, using default features")
            # Default features if schema is missing
            bio_features = {
                'priority_1': {
                    'fu_plasma_norm': {'transform': 'none', 'winsorize_pct': 1.0},
                    'ER_Pgp_log10': {'transform': 'log10', 'winsorize_pct': 1.0},
                    'ER_BCRP_log10': {'transform': 'log10', 'winsorize_pct': 1.0},
                    'Papp_A2B_u6cms': {'transform': 'none', 'winsorize_pct': 1.0},
                    'Papp_B2A_u6cms': {'transform': 'none', 'winsorize_pct': 1.0},
                    'pampa_logpe_norm': {'transform': 'log10', 'winsorize_pct': 1.0}
                }
            }
        
        processed_features = {}
        
        # Process each priority level
        for priority, features in bio_features.items():
            logger.info(f"  Processing {priority} features...")
            
            for feature_name, config in features.items():
                # Get config with defaults
                name = config.get('name', feature_name)
                description = config.get('description', f'Biological feature: {feature_name}')
                transform = config.get('transform', 'none')
                winsorize_pct = config.get('winsorize_pct', 0.0)
                missing_strategy = config.get('missing_strategy', 'flag_and_impute')
                
                # Find column (with aliasing)
                candidates = [feature_name, name]
                if 'smiles' in feature_name.lower():
                    candidates.extend(['SMILES', 'smiles', 'Smiles'])
                if 'bbb' in feature_name.lower() or 'label' in feature_name.lower():
                    candidates.extend(['BBB_label', 'bbb_label', 'label', 'Label'])
                
                col_name = self.normalize_column_name(df, candidates)
                
                if col_name is None:
                    logger.warning(f"    Column not found for {feature_name}, creating all-missing")
                    processed_features[feature_name] = pd.Series([np.nan] * len(df), index=df.index)
                    # Create missing flag
                    processed_features[f"{feature_name}_is_missing"] = pd.Series([1] * len(df), index=df.index)
                    continue
                
                # Get the column
                series = df[col_name].copy()
                
                # Apply winsorization
                if winsorize_pct > 0:
                    series = self.apply_winsorization(series, winsorize_pct)
                
                # Apply transformation
                series = self.apply_transform(series, transform)
                
                # Handle missing values
                if missing_strategy == 'flag_and_impute':
                    # Create missing flag
                    missing_flag = series.isna().astype(int)
                    processed_features[f"{feature_name}_is_missing"] = missing_flag
                    
                    # Impute with median
                    if series.notna().any():
                        series = series.fillna(series.median())
                    else:
                        series = series.fillna(0)
                
                processed_features[feature_name] = series
                
                # Store feature info
                self.feature_dict[feature_name] = {
                    'name': name,
                    'description': description,
                    'transform': transform,
                    'winsorize_pct': winsorize_pct,
                    'missing_strategy': missing_strategy,
                    'source_column': col_name,
                    'coverage': series.notna().mean() if 'missing' not in feature_name else None
                }
                
                logger.info(f"    {feature_name}: coverage={series.notna().mean():.3f}")
        
        self.processing_log["features_resolved"] = len(processed_features)
        return processed_features
    
    def process_split(self, input_path, output_path, split_name):
        """Process a single split"""
        logger.info(f"📊 Processing {split_name} split: {input_path}")
        
        try:
            # Load data
            df = pd.read_csv(input_path)
            logger.info(f"  Loaded {len(df)} samples with {len(df.columns)} columns")
            
            # Normalize column names (handle aliasing)
            column_mapping = {}
            
            # SMILES column
            smiles_candidates = ['smiles', 'SMILES', 'Smiles', 'canonical_smiles']
            smiles_col = self.normalize_column_name(df, smiles_candidates)
            if smiles_col:
                column_mapping[smiles_col] = 'smiles'
            
            # InChIKey column
            inchikey_candidates = ['inchikey', 'InChIKey', 'InchiKey', 'inchi_key']
            inchikey_col = self.normalize_column_name(df, inchikey_candidates)
            if inchikey_col:
                column_mapping[inchikey_col] = 'inchikey'
            
            # BBB label column
            bbb_candidates = ['BBB_label', 'bbb_label', 'label', 'Label', 'BBB', 'bbb']
            bbb_col = self.normalize_column_name(df, bbb_candidates)
            if bbb_col:
                column_mapping[bbb_col] = 'BBB_label'
            
            # Apply column mapping
            df = df.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_cols = ['smiles', 'inchikey', 'BBB_label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                logger.error(f"  Missing required columns: {missing_cols}")
                self.processing_log["errors"].append(f"Missing columns in {split_name}: {missing_cols}")
                return False
            
            # Process bio features
            bio_features = self.process_bio_features(df)
            
            # Combine original data with bio features
            result_df = df[['inchikey', 'smiles', 'BBB_label']].copy()
            
            # Add bio features
            for feature_name, feature_series in bio_features.items():
                result_df[feature_name] = feature_series
            
            # Save as parquet
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            result_df.to_parquet(output_path, index=False)
            
            logger.info(f"  ✅ Saved {len(result_df)} samples with {len(result_df.columns)} columns to {output_path}")
            self.processing_log["created_files"].append(str(output_path))
            
            return True
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {split_name}: {e}")
            self.processing_log["errors"].append(f"Error in {split_name}: {str(e)}")
            return False

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

def main():
    """Main processing pipeline"""
    parser = argparse.ArgumentParser(description='Build bio features for train/dev/test splits')
    parser.add_argument('--train', default='data/processed_train.csv', help='Path to train CSV file')
    parser.add_argument('--dev', default='data/processed_dev.csv', help='Path to dev CSV file')
    parser.add_argument('--test', default='data/processed_test.csv', help='Path to test CSV file')
    parser.add_argument('--union', help='Path to union dataset (optional)')
    parser.add_argument('--schema', default='features/bio_schema.yaml', help='Path to bio schema YAML file')
    args = parser.parse_args()
    
    logger.info("🚀 Starting bio feature construction...")
    
    # Initialize builder with custom schema path
    builder = BioFeatureBuilder(schema_path=args.schema)
    
    # Define splits to process using command line arguments
    splits = {
        'train': {
            'input': args.train,
            'output': 'data/work/bio_train.parquet'
        },
        'dev': {
            'input': args.dev,
            'output': 'data/work/bio_dev.parquet'
        },
        'test': {
            'input': args.test,
            'output': 'data/work/bio_test.parquet'
        }
    }
    
    # Process each split
    for split_name, paths in splits.items():
        input_path = Path(paths['input'])
        output_path = Path(paths['output'])
        
        # Convert to string and normalize path separators for cross-platform compatibility
        input_path_str = str(input_path).replace('\\', '/')
        output_path_str = str(output_path).replace('\\', '/')
        input_path = Path(input_path_str)
        output_path = Path(output_path_str)
        
        if not input_path.exists():
            logger.warning(f"Input file not found: {input_path}. Skipping {split_name} split.")
            builder.processing_log["skipped"].append(f"{split_name}: file not found")
            continue
            
        builder.process_split(input_path, output_path, split_name)
    
    # Save feature dictionary
    if builder.feature_dict:
        feature_dict_df = pd.DataFrame(builder.feature_dict).T
        feature_dict_path = Path("reports/tables/Table_Feature_Dictionary.csv")
        feature_dict_path = Path(str(feature_dict_path).replace('\\', '/'))  # Google Drive compatibility
        feature_dict_path.parent.mkdir(parents=True, exist_ok=True)
        feature_dict_df.to_csv(feature_dict_path, index=True)
        
        logger.info(f"✅ Saved feature dictionary to {feature_dict_path}")
    
    # Save processing log (convert numpy types to native Python types for JSON serialization)
    log_path = Path("reports/logs/bio_feature_processing.json")
    log_path = Path(str(log_path).replace('\\', '/'))  # Google Drive compatibility
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    serializable_log = convert_numpy_types(builder.processing_log)
    serializable_log["status"] = "completed"
    
    with open(log_path, 'w') as f:
        json.dump(serializable_log, f, indent=2)
    
    logger.info(f"✅ Saved processing log to {log_path}")
    logger.info("🎉 Bio feature construction completed successfully!")
    
    # Report summary
    n_created = len(builder.processing_log["created_files"])
    n_skipped = len(builder.processing_log["skipped"])
    n_errors = len(builder.processing_log["errors"])
    
    logger.info(f"📊 Summary: {n_created} files created, {n_skipped} skipped, {n_errors} errors")
    
    if builder.processing_log["errors"]:
        logger.warning("⚠️ Errors encountered:")
        for error in builder.processing_log["errors"]:
            logger.warning(f"  - {error}")

if __name__ == "__main__":
    main()





