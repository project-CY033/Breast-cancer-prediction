import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import os

# --- 1. Synthetic NGS Dataset Generation ---
def generate_synthetic_data(n_samples=5000):
    np.random.seed(42)
    
    # Genetic features (Mutations in key genes: 0 = Wild-type, 1 = Mutated)
    data = {
        'BRCA1_Mutation': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        'BRCA2_Mutation': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        'TP53_Mutation': np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2]),
        'HER2_Amplification': np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15]),
        'PIK3CA_Mutation': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'Mean_Sequence_Quality': np.random.normal(loc=32, scale=5, size=n_samples).clip(20, 40),
        'BRCA1_Impact_Score': np.random.uniform(0, 1, size=n_samples),
        'BRCA2_Impact_Score': np.random.uniform(0, 1, size=n_samples),
        'TP53_Impact_Score': np.random.uniform(0, 1, size=n_samples),
        'Total_Variants_Found': np.random.poisson(lam=15, size=n_samples),
        'Age_at_Diagnosis': np.random.randint(30, 85, size=n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Base risk calculation
    risk = (
        df['BRCA1_Mutation'] * 2.5 * df['BRCA1_Impact_Score'] +
        df['BRCA2_Mutation'] * 2.0 * df['BRCA2_Impact_Score'] +
        df['TP53_Mutation'] * 1.5 * df['TP53_Impact_Score'] +
        df['HER2_Amplification'] * 1.2 +
        df['PIK3CA_Mutation'] * 0.8 +
        (df['Age_at_Diagnosis'] - 30) * 0.02 + 
        np.random.normal(0, 0.5, size=n_samples)
    )
    
    prob = 1 / (1 + np.exp(-(risk - 1.5)))
    df['Diagnosis_Cancer'] = (prob > 0.5).astype(int)
    
    df.to_csv("synthetic_ngs_breast_cancer.csv", index=False)
    print(f"Dataset generated and saved. Shape: {df.shape}")
    return df

# --- 2. Build the Advanced Model Pipeline & Comparative Models ---
def train_and_save_model(df):
    X = df.drop(columns=['Diagnosis_Cancer'])
    y = df['Diagnosis_Cancer']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ---------------------------------------------------------
    # Models exactly as specified in the research report
    # ---------------------------------------------------------
    models_dict = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced'),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Artificial Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True),
        "XGradient Boosting": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, use_label_encoder=False, eval_metric='logloss')
    }
    
    metrics_results = {}
    fitted_models = {}

    for name, clf in models_dict.items():
        print(f"Training {name}...")
        clf.fit(X_train_scaled, y_train)
        fitted_models[name] = clf
        
        y_pred = clf.predict(X_test_scaled)
        y_pred_prob = clf.predict_proba(X_test_scaled)[:, 1]
        
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        
        metrics_results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_prob),
            'cm': cm.tolist(),
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist()
        }
        
    # --- Advanced Upgrade: Voting Classifier ---
    print("Building Ultimate Voting Classifier Ensemble...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', fitted_models["Random Forest"]),
            ('xgb', fitted_models["XGradient Boosting"]),
            ('mlp', fitted_models["Artificial Neural Network (MLP)"])
        ],
        voting='soft'
    )
    ensemble.fit(X_train_scaled, y_train)

    # Save objects needed for UI
    model_data = {
        'model': ensemble,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'xgb_for_shap': fitted_models["XGradient Boosting"],
        'comparative_metrics': metrics_results # New feature tying back to the report Chapter 4
    }
    
    joblib.dump(model_data, "advanced_breast_cancer_model.pkl")
    print("Model pipeline and comparative metrics saved to advanced_breast_cancer_model.pkl")

if __name__ == "__main__":
    if not os.path.exists("synthetic_ngs_breast_cancer.csv"):
        print("Generating Data...")
        df = generate_synthetic_data(10000)
    else:
        print("Loading Data...")
        df = pd.read_csv("synthetic_ngs_breast_cancer.csv")
        
    train_and_save_model(df)
