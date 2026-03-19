import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12


class Visualizer:
    def __init__(self, output_dir: str = "output/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str] = None,
                             title: str = "Confusion Matrix",
                             save_path: str = None) -> plt.Figure:
        if labels is None:
            labels = ["Healthy", "Breast Cancer"]
            
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=labels, yticklabels=labels, ax=ax,
                   cbar_kws={"label": "Count"})
        
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
            
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       title: str = "ROC Curve", save_path: str = None) -> plt.Figure:
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, color="darkorange", lw=2,
                label=f"ROC Curve (AUC = {roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", 
                label="Random Classifier")
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"ROC curve saved to {save_path}")
            
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                    title: str = "Precision-Recall Curve",
                                    save_path: str = None) -> plt.Figure:
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        ap = average_precision_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, color="blue", lw=2,
                label=f"PR Curve (AP = {ap:.3f})")
        
        baseline = y_true.sum() / len(y_true)
        ax.axhline(y=baseline, color="red", linestyle="--", 
                   label=f"Baseline (AP = {baseline:.3f})")
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(title)
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Precision-Recall curve saved to {save_path}")
            
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, 
                                top_n: int = 20,
                                title: str = "Feature Importance",
                                save_path: str = None) -> plt.Figure:
        if importance_df.empty:
            logger.warning("Empty importance DataFrame")
            return None
            
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(top_features)))
        
        ax.barh(range(len(top_features)), top_features["importance"].values, 
                color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features["feature"].values)
        ax.invert_yaxis()
        
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        ax.set_title(f"Top {top_n} {title}")
        ax.grid(True, axis="x", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Feature importance plot saved to {save_path}")
            
        return fig
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                             metrics: List[str] = None,
                             title: str = "Model Comparison",
                             save_path: str = None) -> plt.Figure:
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
            
        comparison_melted = comparison_df.melt(
            id_vars=["model"], 
            value_vars=metrics,
            var_name="Metric", 
            value_name="Score"
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.barplot(data=comparison_melted, x="model", y="Score", hue="Metric", ax=ax)
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(title="Metrics", bbox_to_anchor=(1.02, 1), loc="upper left")
        ax.set_ylim([0, 1.1])
        ax.grid(True, axis="y", alpha=0.3)
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Model comparison plot saved to {save_path}")
            
        return fig
    
    def plot_gene_mutation_distribution(self, df: pd.DataFrame,
                                       gene_col: str = "gene",
                                       title: str = "Gene Mutation Distribution",
                                       save_path: str = None) -> plt.Figure:
        if gene_col not in df.columns:
            logger.error(f"Column {gene_col} not found")
            return None
            
        mutation_counts = df[gene_col].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(mutation_counts)))
        
        axes[0].pie(mutation_counts.values, labels=mutation_counts.index, 
                   autopct="%1.1f%%", colors=colors, startangle=90)
        axes[0].set_title(f"{title} - Pie Chart")
        
        axes[1].bar(mutation_counts.index, mutation_counts.values, 
                   color=colors, edgecolor="black")
        axes[0].set_title(f"{title} - Bar Chart")
        
        for ax in axes:
            ax.tick_params(axis="x", rotation=45)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Gene mutation distribution saved to {save_path}")
            
        return fig
    
    def plot_mutation_type_distribution(self, df: pd.DataFrame,
                                       mutation_col: str = "mutation_type",
                                       label_col: str = "label",
                                       title: str = "Mutation Types by Class",
                                       save_path: str = None) -> plt.Figure:
        if mutation_col not in df.columns or label_col not in df.columns:
            logger.error("Required columns not found")
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        crosstab = pd.crosstab(df[mutation_col], df[label_col])
        crosstab.plot(kind="bar", ax=ax, color=["#2ecc71", "#e74c3c"], edgecolor="black")
        
        ax.set_xlabel("Mutation Type")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(["Healthy", "Breast Cancer"], title="Class")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y", alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Mutation type distribution saved to {save_path}")
            
        return fig
    
    def plot_pathway_scores(self, df: pd.DataFrame,
                           score_cols: List[str] = None,
                           title: str = "Pathway Dysregulation Scores",
                           save_path: str = None) -> plt.Figure:
        if score_cols is None:
            score_cols = ["brca_pathway_score", "pi3k_pathway_score", 
                         "emt_pathway_score", "hormone_pathway_score"]
            
        valid_cols = [c for c in score_cols if c in df.columns]
        
        if not valid_cols:
            logger.error("No valid pathway score columns found")
            return None
            
        fig, ax = plt.subplots(figsize=(12, 6))
        
        pathway_means = df[valid_cols].mean()
        pathway_stds = df[valid_cols].std()
        
        colors = plt.cm.Paired(np.linspace(0, 1, len(valid_cols)))
        
        ax.bar(pathway_means.index, pathway_means.values, 
               yerr=pathway_stds.values, capsize=5,
               color=colors, edgecolor="black", alpha=0.8)
        
        ax.set_xlabel("Pathway")
        ax.set_ylabel("Mean Score")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        plt.xticks(rotation=45, ha="right")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Pathway scores plot saved to {save_path}")
            
        return fig
    
    def create_dashboard(self, metrics: Dict, y_true: np.ndarray, y_pred_proba: np.ndarray,
                        importance_df: pd.DataFrame, save_path: str = None) -> plt.Figure:
        fig = plt.figure(figsize=(16, 12))
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, :2])
        ax5 = fig.add_subplot(gs[1, 2])
        ax6 = fig.add_subplot(gs[2, :])
        
        cm = confusion_matrix(y_true, (y_pred_proba > 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                   xticklabels=["H", "BC"], yticklabels=["H", "BC"])
        ax1.set_title("Confusion Matrix")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax2.plot(fpr, tpr, color="darkorange", lw=2)
        ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax2.set_title(f"ROC Curve (AUC={roc_auc:.3f})")
        ax2.set_xlabel("FPR")
        ax2.set_ylabel("TPR")
        ax2.grid(True, alpha=0.3)
        
        metrics_text = f"""Model Performance Summary
        Accuracy: {metrics['accuracy']:.4f}
        Precision: {metrics['precision']:.4f}
        Recall: {metrics['recall']:.4f}
        F1-Score: {metrics['f1_score']:.4f}
        ROC-AUC: {metrics['roc_auc']:.4f}
        """
        ax3.text(0.1, 0.5, metrics_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment="center",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax3.axis("off")
        ax3.set_title("Performance Metrics")
        
        top_15 = importance_df.head(15)
        ax4.barh(range(len(top_15)), top_15["importance"].values, 
                color=plt.cm.viridis(np.linspace(0, 0.8, len(top_15))))
        ax4.set_yticks(range(len(top_15)))
        ax4.set_yticklabels(top_15["feature"].values, fontsize=9)
        ax4.invert_yaxis()
        ax4.set_xlabel("Importance")
        ax4.set_title("Top 15 Feature Importance")
        ax4.grid(True, axis="x", alpha=0.3)
        
        sample_counts = [len(y_true[y_true == 0]), len(y_true[y_true == 1])]
        ax5.pie(sample_counts, labels=["Healthy", "Breast Cancer"],
               autopct="%1.1f%%", colors=["#2ecc71", "#e74c3c"],
               explode=[0.05, 0.05], startangle=90)
        ax5.set_title("Class Distribution")
        
        cv_scores = metrics.get("cross_val_scores", [])
        if cv_scores:
            ax6.plot(range(1, len(cv_scores) + 1), cv_scores, 
                     marker="o", markersize=10, linewidth=2, color="blue")
            ax6.axhline(y=np.mean(cv_scores), color="red", linestyle="--",
                       label=f"Mean: {np.mean(cv_scores):.4f}")
            ax6.fill_between(range(1, len(cv_scores) + 1), cv_scores,
                           alpha=0.3, color="blue")
            ax6.set_xlabel("Fold")
            ax6.set_ylabel("Accuracy")
            ax6.set_title("Cross-Validation Results")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            ax6.set_xticks(range(1, len(cv_scores) + 1))
        
        fig.suptitle("Breast Cancer Mutation Prediction - Results Dashboard",
                    fontsize=16, fontweight="bold", y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Dashboard saved to {save_path}")
            
        return fig


if __name__ == "__main__":
    pass
