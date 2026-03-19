import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: np.ndarray
    cross_val_scores: List[float]
    
    def to_dict(self) -> Dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "cross_val_scores": self.cross_val_scores
        }


class ModelTrainer:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def get_model(self, model_type: str, **kwargs) -> Any:
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 10),
                min_samples_split=kwargs.get("min_samples_split", 5),
                random_state=kwargs.get("random_state", 42),
                n_jobs=-1
            ),
            "svm": SVC(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 1.0),
                gamma=kwargs.get("gamma", "scale"),
                probability=True,
                random_state=kwargs.get("random_state", 42)
            ),
            "logistic_regression": LogisticRegression(
                max_iter=kwargs.get("max_iter", 1000),
                C=kwargs.get("C", 1.0),
                random_state=kwargs.get("random_state", 42)
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=kwargs.get("n_estimators", 100),
                max_depth=kwargs.get("max_depth", 5),
                learning_rate=kwargs.get("learning_rate", 0.1),
                random_state=kwargs.get("random_state", 42)
            ),
            "neural_network": MLPClassifier(
                hidden_layer_sizes=kwargs.get("hidden_layers", (100, 50)),
                max_iter=kwargs.get("max_iter", 500),
                alpha=kwargs.get("alpha", 0.001),
                random_state=kwargs.get("random_state", 42)
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return models[model_type]
    
    def train_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series, **kwargs) -> Tuple[Any, ModelMetrics]:
        
        logger.info(f"Training {model_type} model...")
        
        model = self.get_model(model_type, **kwargs)
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            confusion_matrix=cm,
            cross_val_scores=cv_scores.tolist()
        )
        
        self.models[model_type] = {
            "model": model,
            "metrics": metrics,
            "feature_columns": X_train.columns.tolist()
        }
        
        if metrics.roc_auc > self.best_score:
            self.best_score = metrics.roc_auc
            self.best_model = model_type
        
        logger.info(f"{model_type} trained - ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}")
        
        return model, metrics
    
    def hyperparameter_tuning(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                              param_grid: Dict, cv: int = 3) -> Tuple[Any, Dict]:
        
        logger.info(f"Hyperparameter tuning for {model_type}...")
        
        model = self.get_model(model_type)
        
        grid_search = GridSearchCV(
            model, param_grid, cv=StratifiedKFold(n_splits=cv),
            scoring="roc_auc", n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def save_model(self, model_type: str, filepath: str = None) -> Path:
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found")
            
        if filepath is None:
            filepath = self.model_dir / f"{model_type}_model.pkl"
        else:
            filepath = Path(filepath)
            
        with open(filepath, "wb") as f:
            pickle.dump(self.models[model_type], f)
            
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    def load_model(self, model_type: str, filepath: str) -> Any:
        with open(filepath, "rb") as f:
            self.models[model_type] = pickle.load(f)
            
        logger.info(f"Model loaded from {filepath}")
        return self.models[model_type]["model"]
    
    def get_predictions(self, model_type: str, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found")
            
        model = self.models[model_type]["model"]
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        return y_pred, y_pred_proba
    
    def get_feature_importance(self, model_type: str) -> pd.DataFrame:
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not found")
            
        model = self.models[model_type]["model"]
        features = self.models[model_type]["feature_columns"]
        
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                "feature": features,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
        elif hasattr(model, "coef_"):
            importance_df = pd.DataFrame({
                "feature": features,
                "importance": np.abs(model.coef_[0])
            }).sort_values("importance", ascending=False)
        else:
            importance_df = pd.DataFrame()
            
        return importance_df
    
    def compare_models(self) -> pd.DataFrame:
        results = []
        
        for model_type, data in self.models.items():
            metrics = data["metrics"]
            results.append({
                "model": model_type,
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "roc_auc": metrics.roc_auc,
                "cv_mean": np.mean(metrics.cross_val_scores),
                "cv_std": np.std(metrics.cross_val_scores)
            })
            
        return pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    
    def get_best_model(self) -> Tuple[str, Any, ModelMetrics]:
        if self.best_model is None:
            raise ValueError("No models trained yet")
            
        data = self.models[self.best_model]
        return self.best_model, data["model"], data["metrics"]


class BreastCancerPredictor:
    def __init__(self, model_dir: str = "models"):
        self.trainer = ModelTrainer(model_dir)
        self.model = None
        self.feature_columns = None
        self.model_type = None
        
    def train(self, df: pd.DataFrame, target_col: str = "label",
              model_type: str = "random_forest", **kwargs) -> ModelMetrics:
        
        feature_cols = [c for c in df.columns if c not in ["sample_id", target_col]]
        X = df[feature_cols]
        y = df[target_col]
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model, metrics = self.trainer.train_model(
            model_type, X_train, y_train, X_test, y_test, **kwargs
        )
        
        self.feature_columns = feature_cols
        self.model_type = model_type
        
        return metrics
    
    def predict(self, sample_data: Dict) -> Dict:
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        df = pd.DataFrame([sample_data])
        df = df[self.feature_columns]
        
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0]
        
        return {
            "prediction": int(prediction),
            "label": "Breast Cancer" if prediction == 1 else "Healthy",
            "probability_cancer": float(probability[1]),
            "probability_healthy": float(probability[0])
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        feature_cols = [c for c in df.columns if c in self.feature_columns]
        X = df[feature_cols]
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        results = df[["sample_id"]].copy() if "sample_id" in df.columns else df.copy()
        results["prediction"] = predictions
        results["label"] = ["Breast Cancer" if p == 1 else "Healthy" for p in predictions]
        results["probability_cancer"] = probabilities
        
        return results
    
    def save(self, filepath: str = "models/breast_cancer_model.pkl"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_columns": self.feature_columns,
                "model_type": self.model_type,
                "trainer": self.trainer
            }, f)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str = "models/breast_cancer_model.pkl"):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_columns = data["feature_columns"]
        self.model_type = data["model_type"]
        self.trainer = data.get("trainer")
        logger.info(f"Model loaded from {filepath}")


def train_models_pipeline(data_path: str, output_dir: str = "models") -> Dict:
    df = pd.read_csv(data_path)
    
    feature_cols = [c for c in df.columns if c not in ["sample_id", "label"]]
    X = df[feature_cols]
    y = df["label"]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    trainer = ModelTrainer(output_dir)
    
    model_types = ["random_forest", "svm", "logistic_regression", "gradient_boosting"]
    results = {}
    
    for model_type in model_types:
        model, metrics = trainer.train_model(
            model_type, X_train, y_train, X_test, y_test
        )
        results[model_type] = metrics.to_dict()
        trainer.save_model(model_type)
    
    comparison = trainer.compare_models()
    best_model = trainer.best_model
    
    return {
        "results": results,
        "comparison": comparison.to_dict(),
        "best_model": best_model
    }


if __name__ == "__main__":
    pass
