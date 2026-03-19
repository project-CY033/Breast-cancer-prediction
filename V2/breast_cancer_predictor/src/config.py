import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

GENES = [
    "BRCA1", "BRCA2", "TP53", "PIK3CA", "PTEN", "CDH1",
    "GATA3", "MAP3K1", "TBX3", "FOXA1", "ESR1", "HER2",
    "MYC", "CCND1", "FGFR1", "FGFR2", "EGFR", "BRAF"
]

MUTATION_TYPES = [
    "missense", "nonsense", "frameshift", "silent", 
    "splice_site", "promoter", "deletion", "insertion"
]

FEATURE_COLUMNS = [
    "sample_id", "label", "total_mutations",
    "brca1_mutations", "brca2_mutations", "tp53_mutations",
    "missense_count", "nonsense_count", "frameshift_count",
    "gene_expression_score", "pathway_score", "functional_impact_score"
]

ML_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
        "n_jobs": -1
    },
    "svm": {
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
        "random_state": 42
    },
    "logistic_regression": {
        "max_iter": 1000,
        "C": 1.0,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42,
        "n_jobs": -1
    }
}
