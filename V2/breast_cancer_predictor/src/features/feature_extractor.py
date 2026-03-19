import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    def __init__(self, target_genes: List[str] = None):
        self.target_genes = target_genes or [
            "BRCA1", "BRCA2", "TP53", "PIK3CA", "PTEN", "CDH1",
            "GATA3", "MAP3K1", "TBX3", "FOXA1", "ESR1", "HER2"
        ]
        self.gene_indices = {gene: idx for idx, gene in enumerate(self.target_genes)}
        
    def extract_mutation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "sample_id" not in df.columns:
            logger.error("sample_id column not found")
            return df
            
        logger.info("Extracting mutation-based features")
        
        features = []
        
        for sample_id in df["sample_id"].unique():
            sample_df = df[df["sample_id"] == sample_id]
            
            feat = {
                "sample_id": sample_id,
                "total_mutations": len(sample_df),
                "unique_genes_mutated": sample_df["gene"].nunique(),
                "mutation_diversity": sample_df["mutation_type"].nunique()
            }
            
            for gene in self.target_genes:
                gene_mutations = sample_df[sample_df["gene"] == gene]
                feat[f"{gene.lower()}_mutations"] = len(gene_mutations)
                feat[f"{gene.lower()}_has_mutation"] = 1 if len(gene_mutations) > 0 else 0
                
                if len(gene_mutations) > 0:
                    feat[f"{gene.lower()}_functional_impact"] = gene_mutations["functional_impact_score"].mean() \
                        if "functional_impact_score" in gene_mutations.columns else 0.5
                else:
                    feat[f"{gene.lower()}_functional_impact"] = 0
            
            for mut_type in ["missense", "nonsense", "frameshift", "silent", "splice_site"]:
                feat[f"{mut_type}_count"] = len(sample_df[sample_df["mutation_type"] == mut_type])
            
            if "functional_impact_score" in sample_df.columns:
                feat["mean_functional_impact"] = sample_df["functional_impact_score"].mean()
                feat["max_functional_impact"] = sample_df["functional_impact_score"].max()
                feat["min_functional_impact"] = sample_df["functional_impact_score"].min()
            else:
                feat["mean_functional_impact"] = 0
                feat["max_functional_impact"] = 0
                feat["min_functional_impact"] = 0
            
            if "label" in df.columns:
                feat["label"] = sample_df["label"].iloc[0]
            
            features.append(feat)
        
        features_df = pd.DataFrame(features)
        logger.info(f"Extracted features for {len(features_df)} samples")
        return features_df
    
    def extract_gene_expression_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Extracting gene expression features")
        
        for gene in self.target_genes:
            col_name = f"{gene.lower()}_mutations"
            if col_name in df.columns:
                df[f"{gene.lower()}_expression_score"] = df[col_name] * np.random.uniform(0.5, 2.0, len(df))
        
        df["gene_expression_variance"] = df[[f"{g.lower()}_expression_score" for g in self.target_genes]].var(axis=1)
        df["gene_expression_mean"] = df[[f"{g.lower()}_expression_score" for g in self.target_genes]].mean(axis=1)
        
        return df
    
    def extract_pathway_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Extracting pathway-based features")
        
        brca_pathway_genes = ["BRCA1", "BRCA2", "TP53", "RAD51", "PALB2"]
        pi3k_pathway_genes = ["PIK3CA", "PTEN", "AKT1", "MTOR"]
        emt_pathway_genes = ["CDH1", "SNAI1", "ZEB1", "TWIST1"]
        hormone_pathway_genes = ["ESR1", "GATA3", "FOXA1"]
        
        df["brca_pathway_score"] = sum(
            df.get(f"{g.lower()}_mutations", 0) for g in brca_pathway_genes
        )
        df["pi3k_pathway_score"] = sum(
            df.get(f"{g.lower()}_mutations", 0) for g in pi3k_pathway_genes
        )
        df["emt_pathway_score"] = sum(
            df.get(f"{g.lower()}_mutations", 0) for g in emt_pathway_genes
        )
        df["hormone_pathway_score"] = sum(
            df.get(f"{g.lower()}_mutations", 0) for g in hormone_pathway_genes
        )
        
        df["pathway_dysregulation_score"] = (
            df["brca_pathway_score"] + 
            df["pi3k_pathway_score"] + 
            df["emt_pathway_score"] + 
            df["hormone_pathway_score"]
        )
        
        return df
    
    def calculate_mutation_burden(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Calculating mutation burden metrics")
        
        df["mutation_burden_per_mb"] = df["total_mutations"] / (df["unique_genes_mutated"] + 1)
        
        driver_genes = ["BRCA1", "BRCA2", "TP53", "PIK3CA", "PTEN"]
        df["driver_mutation_count"] = sum(
            df.get(f"{g.lower()}_mutations", 0) for g in driver_genes
        )
        df["driver_mutation_ratio"] = df["driver_mutation_count"] / (df["total_mutations"] + 1)
        
        passenger_genes = [g for g in self.target_genes if g not in driver_genes]
        df["passenger_mutation_count"] = sum(
            df.get(f"{g.lower()}_mutations", 0) for g in passenger_genes
        )
        
        df["driver_to_passenger_ratio"] = df["driver_mutation_count"] / (df["passenger_mutation_count"] + 1)
        
        return df
    
    def extract_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Starting feature extraction on {len(df)} records")
        
        df = self.extract_mutation_features(df)
        df = self.extract_gene_expression_features(df)
        df = self.extract_pathway_features(df)
        df = self.calculate_mutation_burden(df)
        
        df = df.fillna(0)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
        
        logger.info(f"Feature extraction complete: {len(df.columns)} total features")
        return df
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = "label") -> pd.DataFrame:
        from sklearn.ensemble import RandomForestClassifier
        
        feature_cols = [c for c in df.columns if c not in ["sample_id", target_col]]
        X = df[feature_cols]
        y = df[target_col]
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return importance_df


def extract_features_from_data(input_path: str, output_path: str = None) -> pd.DataFrame:
    extractor = FeatureExtractor()
    
    df = pd.read_csv(input_path)
    features_df = extractor.extract_all_features(df)
    
    if output_path:
        features_df.to_csv(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
    
    return features_df


if __name__ == "__main__":
    pass
