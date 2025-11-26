import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, f1_score, precision_score, 
                             recall_score, fbeta_score)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnsembleFraudDetector:
    def __init__(self, catboost_path='models/catboost_model.cbm',
                 nn_path='models/nn_model.h5',
                 scaler_path='models/nn_scaler.pkl'):
        
        print("loading models...")
        
        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model(catboost_path)
        
        self.nn_model = keras.models.load_model(nn_path)
        self.scaler = joblib.load(scaler_path)
        
        self.best_threshold = 0.5
        self.weights = None
        
        print("ready")
    
    def predict_ensemble(self, X, method='weighted_average', weights=None):
        X_clean = X.copy()
        for col in X_clean.columns:
            if not pd.api.types.is_numeric_dtype(X_clean[col]):
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        X_clean = X_clean.fillna(X_clean.median())
        
        catboost_proba = self.catboost_model.predict_proba(X_clean)[:, 1]
        X_scaled = self.scaler.transform(X_clean)
        nn_proba = self.nn_model.predict(X_scaled, verbose=0).flatten()
        
        if method == 'weighted_average':
            weights = weights or self.weights or [0.5, 0.5]
            ensemble_proba = weights[0] * catboost_proba + weights[1] * nn_proba
        elif method == 'voting':
            catboost_pred = (catboost_proba >= 0.5).astype(int)
            nn_pred = (nn_proba >= 0.5).astype(int)
            ensemble_pred = ((catboost_pred + nn_pred) >= 1).astype(int)
            ensemble_proba = (catboost_proba + nn_proba) / 2
        else:
            raise ValueError(f"unknown method: {method}")
        
        return ensemble_proba
    
    def optimize_weights(self, X_val, y_val, beta=2):
        print("optimizing weights...")
        
        best_fbeta = 0
        best_weights = [0.5, 0.5]
        best_threshold = 0.5
        
        for w_catboost in np.arange(0.0, 1.1, 0.1):
            w_nn = 1.0 - w_catboost
            weights = [w_catboost, w_nn]
            ensemble_proba = self.predict_ensemble(X_val, method='weighted_average', weights=weights)
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                ensemble_pred = (ensemble_proba >= threshold).astype(int)
                fbeta = fbeta_score(y_val, ensemble_pred, beta=beta)
                
                if fbeta > best_fbeta:
                    best_fbeta = fbeta
                    best_weights = weights
                    best_threshold = threshold
        
        self.weights = best_weights
        self.best_threshold = best_threshold
        
        print(f"weights: catboost={best_weights[0]:.2f}, nn={best_weights[1]:.2f}")
        print(f"threshold: {best_threshold:.2f}, fbeta: {best_fbeta:.4f}")
        
        return best_weights, best_threshold
    
    def evaluate(self, X_test, y_test):
        print("evaluating ensemble...")
        
        ensemble_proba = self.predict_ensemble(X_test, method='weighted_average', weights=self.weights)
        ensemble_pred = (ensemble_proba >= self.best_threshold).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, ensemble_proba),
            'f1': f1_score(y_test, ensemble_pred),
            'f2': fbeta_score(y_test, ensemble_pred, beta=2),
            'precision': precision_score(y_test, ensemble_pred),
            'recall': recall_score(y_test, ensemble_pred)
        }
        
        print(f"roc_auc: {metrics['roc_auc']:.4f}")
        print(f"f1: {metrics['f1']:.4f}, f2: {metrics['f2']:.4f}")
        print(f"precision: {metrics['precision']:.4f}, recall: {metrics['recall']:.4f}")
        print(classification_report(y_test, ensemble_pred, target_names=['Clean', 'Fraud']))
        
        cm = confusion_matrix(y_test, ensemble_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"tp: {tp}, fn: {fn}, fp: {fp}, tn: {tn}")
        fraud_catch_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"fraud_catch_rate: {fraud_catch_rate:.2%}")
        print(f"false_positive_rate: {false_positive_rate:.2%}")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
                   xticklabels=['Clean', 'Fraud'],
                   yticklabels=['Clean', 'Fraud'])
        plt.title('Confusion Matrix - Ensemble', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('outputs/confusion_matrix_ensemble.png', dpi=300, bbox_inches='tight')
        print("saved: outputs/confusion_matrix_ensemble.png")
        
        metrics.update({
            'fraud_catch_rate': fraud_catch_rate,
            'false_positive_rate': false_positive_rate,
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        })
        
        return metrics
    
    def compare_models(self, X_test, y_test):
        print("comparing models...")
        
        X_clean = X_test.copy()
        for col in X_clean.columns:
            if not pd.api.types.is_numeric_dtype(X_clean[col]):
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        X_clean = X_clean.fillna(X_clean.median())
        
        catboost_proba = self.catboost_model.predict_proba(X_clean)[:, 1]
        catboost_pred = (catboost_proba >= 0.5).astype(int)
        
        X_scaled = self.scaler.transform(X_clean)
        nn_proba = self.nn_model.predict(X_scaled, verbose=0).flatten()
        nn_pred = (nn_proba >= 0.5).astype(int)
        
        ensemble_proba = self.predict_ensemble(X_test, method='weighted_average', weights=self.weights)
        ensemble_pred = (ensemble_proba >= self.best_threshold).astype(int)
        
        comparison = pd.DataFrame({
            'Model': ['CatBoost', 'Neural Network', 'Ensemble'],
            'ROC-AUC': [
                roc_auc_score(y_test, catboost_proba),
                roc_auc_score(y_test, nn_proba),
                roc_auc_score(y_test, ensemble_proba)
            ],
            'F1-Score': [
                f1_score(y_test, catboost_pred),
                f1_score(y_test, nn_pred),
                f1_score(y_test, ensemble_pred)
            ],
            'F2-Score': [
                fbeta_score(y_test, catboost_pred, beta=2),
                fbeta_score(y_test, nn_pred, beta=2),
                fbeta_score(y_test, ensemble_pred, beta=2)
            ],
            'Precision': [
                precision_score(y_test, catboost_pred),
                precision_score(y_test, nn_pred),
                precision_score(y_test, ensemble_pred)
            ],
            'Recall': [
                recall_score(y_test, catboost_pred),
                recall_score(y_test, nn_pred),
                recall_score(y_test, ensemble_pred)
            ]
        })
        
        print(comparison.to_string(index=False))
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        metrics = ['ROC-AUC', 'F1-Score', 'F2-Score', 'Precision', 'Recall']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        x = np.arange(len(comparison))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            axes[0].bar(x + i * width, comparison[metric], width, label=metric, color=colors[i])
        
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xticks(x + width * 2)
        axes[0].set_xticklabels(comparison['Model'])
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        best_counts = comparison[metrics].apply(lambda x: (x == x.max()).sum(), axis=1)
        axes[1].bar(comparison['Model'], best_counts, color=['teal', 'coral', 'purple'])
        axes[1].set_ylabel('Number of Best Metrics')
        axes[1].set_title('Winner Count by Metrics', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('outputs/model_comparison.png', dpi=300, bbox_inches='tight')
        print("saved: outputs/model_comparison.png")
        
        comparison.to_csv('outputs/model_comparison.csv', index=False)
        print("saved: outputs/model_comparison.csv")
        
        return comparison
    
    def save_ensemble(self, path='models/ensemble_config.pkl'):
        config = {
            'weights': self.weights,
            'best_threshold': self.best_threshold
        }
        joblib.dump(config, path)
        print(f"saved: {path}")


if __name__ == "__main__":
    data = pd.read_csv('data/processed_features.csv')
    
    with open('data/feature_list.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    X = data[features]
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    ensemble = EnsembleFraudDetector()
    ensemble.optimize_weights(X_val, y_val, beta=2)
    metrics = ensemble.evaluate(X_test, y_test)
    comparison = ensemble.compare_models(X_test, y_test)
    ensemble.save_ensemble()
    
    pd.DataFrame([metrics]).to_csv('outputs/ensemble_metrics.csv', index=False)
    print("done")

