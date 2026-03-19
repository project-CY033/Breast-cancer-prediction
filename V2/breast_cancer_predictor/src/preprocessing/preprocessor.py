import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self):
        self.scaler_params = {}
        self.feature_means = {}
        self.feature_medians = {}
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()
        initial_len = len(df_clean)
        
        df_clean = df_clean.drop_duplicates()
        df_clean = df_clean.dropna(subset=["sample_id", "gene", "mutation_type"])
        
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            median = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median)
            self.feature_medians[col] = median
        
        removed = initial_len - len(df_clean)
        logger.info(f"Cleaned data: removed {removed} rows, {len(df_clean)} remaining")
        return df_clean
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        
        mutation_mapping = {
            "missense": 1, "nonsense": 2, "frameshift": 3,
            "silent": 0, "splice_site": 4, "promoter": 5,
            "deletion": 6, "insertion": 7
        }
        
        if "mutation_type" in df_encoded.columns:
            df_encoded["mutation_type_encoded"] = df_encoded["mutation_type"].map(
                lambda x: mutation_mapping.get(x, 0)
            )
        
        gene_dummies = pd.get_dummies(df_encoded["gene"], prefix="gene")
        df_encoded = pd.concat([df_encoded, gene_dummies], axis=1)
        
        logger.info(f"Encoded categorical features: {len(gene_dummies.columns)} gene features added")
        return df_encoded
    
    def handle_outliers(self, df: pd.DataFrame, columns: List[str], 
                        method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
        df_clean = df.copy()
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            if method == "iqr":
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                outliers = (df_clean[col] < lower) | (df_clean[col] > upper)
                df_clean.loc[outliers, col] = df_clean[col].median()
                
            elif method == "zscore":
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean.loc[z_scores > threshold, col] = df_clean[col].median()
        
        return df_clean
    
    def normalize_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_norm = df.copy()
        
        for col in columns:
            if col not in df_norm.columns:
                continue
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            if max_val > min_val:
                df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
                self.scaler_params[col] = {"min": min_val, "max": max_val}
        
        return df_norm
    
    def standardize_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df_std = df.copy()
        
        for col in columns:
            if col not in df_std.columns:
                continue
            mean = df_std[col].mean()
            std = df_std[col].std()
            if std > 0:
                df_std[col] = (df_std[col] - mean) / std
                self.feature_means[col] = mean
                self.scaler_params[col] = {"mean": mean, "std": std}
        
        return df_std
    
    def balance_dataset(self, df: pd.DataFrame, target_col: str = "label",
                       method: str = "smote") -> pd.DataFrame:
        try:
            from imblearn.over_sampling import SMOTE, RandomOverSampler
            from imblearn.under_sampling import RandomUnderSampler
            
            if method == "smote":
                sampler = SMOTE(random_state=42)
            elif method == "oversample":
                sampler = RandomOverSampler(random_state=42)
            elif method == "undersample":
                sampler = RandomUnderSampler(random_state=42)
            else:
                logger.warning(f"Unknown balancing method: {method}")
                return df
            
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            df_balanced = pd.concat([X_resampled, y_resampled], axis=1)
            logger.info(f"Balanced dataset: {len(df)} -> {len(df_balanced)} samples")
            return df_balanced
            
        except ImportError:
            logger.warning("imblearn not installed, skipping balancing")
            return df
    
    def split_data(self, df: pd.DataFrame, target_col: str = "label",
                   test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame]:
        from sklearn.model_selection import train_test_split
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"Split data: train={len(df_train)}, test={len(df_test)}")
        return df_train, df_test
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_col: str = "label",
                           balance: bool = True) -> Tuple[pd.DataFrame]:
        logger.info(f"Starting preprocessing pipeline on {len(df)} samples")
        
        df = self.clean_data(df)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        df = self.handle_outliers(df, numeric_cols)
        df = self.normalize_features(df, numeric_cols)
        
        df = self.encode_categorical(df)
        
        if balance and target_col in df.columns:
            df = self.balance_dataset(df, target_col)
        
        logger.info(f"Preprocessing complete: {len(df)} samples, {len(df.columns)} features")
        return df


def preprocess_data(input_path: str, output_dir: str = "data/processed") -> Tuple[pd.DataFrame]:
    preprocessor = DataPreprocessor()
    
    df = pd.read_csv(input_path)
    df_processed = preprocessor.preprocess_pipeline(df)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df, test_df = preprocessor.split_data(df_processed)
    
    train_path = output_dir / "train_data.csv"
    test_path = output_dir / "test_data.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Preprocessed data saved: {train_path}, {test_path}")
    
    return train_df, test_df


if __name__ == "__main__":
    pass
