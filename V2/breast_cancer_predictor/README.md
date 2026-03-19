# Breast Cancer Mutation Prediction System

A machine learning tool for predicting key mutations in breast cancer based on genomic data.

## Features

- **Data Collection**: Fetch data from TCGA, cBioPortal, NCBI, or load custom files
- **Preprocessing**: Clean, normalize, balance, and prepare data for ML
- **Feature Extraction**: Extract mutation-based, gene expression, and pathway features
- **Multiple ML Models**: Random Forest, SVM, Logistic Regression, Gradient Boosting, Neural Network
- **Visualization**: Confusion matrices, ROC curves, feature importance, dashboards
- **GUI Application**: User-friendly interface for all operations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### GUI Application
```bash
python run.py
```

### Command Line
```bash
python -m src.data_collection.data_collector
python -m src.preprocessing.preprocessor
python -m src.ml_models.model_trainer
```

## Target Genes

BRCA1, BRCA2, TP53, PIK3CA, PTEN, CDH1, GATA3, MAP3K1, TBX3, FOXA1, ESR1, HER2, MYC, CCND1, FGFR1, FGFR2, EGFR, BRAF

## Pathways Analyzed

- BRCA DNA Repair Pathway
- PI3K/AKT Pathway
- EMT Pathway
- Hormone Signaling Pathway

## Project Structure

```
breast_cancer_predictor/
├── config/           # Configuration files
├── data/            # Data storage
├── models/          # Trained models
├── src/
│   ├── data_collection/   # Data fetching
│   ├── preprocessing/     # Data cleaning
│   ├── features/          # Feature extraction
│   ├── ml_models/         # ML training
│   ├── visualization/      # Plots & charts
│   └── gui/              # GUI application
├── tests/           # Unit tests
└── docs/            # Documentation
```

## Research Basis

This tool is based on research from:
1. Breast Cancer Prediction Pipeline (NGS data analysis)
2. Development of Novel Tool for the Prediction of Key Mutations in Breast Cancer Using Machine Learning

## License

MIT License
