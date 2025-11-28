import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, f1_score, precision_score, 
                             recall_score, fbeta_score)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CatBoostFraudDetector:
    def __init__(self, features, target='target'):
        self.features = features
        self.target = target
        self.model = None
        self.best_threshold = 0.5
        
    def prepare_data(self, data, test_size=0.2, random_state=42):
        print("preparing data...")
        X = data[self.features].copy()
        
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = X.fillna(X.median())
        y = data[self.target]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"train: {X_train.shape}, test: {X_test.shape}")
        print(f"train_fraud_rate: {y_train.mean():.4f}, test_fraud_rate: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, X_val, y_val):
        print("training catboost...")
        
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': 100,
            'early_stopping_rounds': 50,
            'auto_class_weights': 'Balanced',
            'task_type': 'CPU'
        }
        
        self.model = CatBoostClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            plot=False
        )
        
        print(f"best_iteration: {self.model.best_iteration_}")
        return self.model
    
    def optimize_threshold(self, X_val, y_val, beta=2):
        print("optimizing threshold...")
        
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.1, 0.9, 0.05)
        
        results = [(t, fbeta_score(y_val, (y_pred_proba >= t).astype(int), beta=beta),
                    precision_score(y_val, (y_pred_proba >= t).astype(int), zero_division=0),
                    recall_score(y_val, (y_pred_proba >= t).astype(int), zero_division=0))
                   for t in thresholds]
        
        best_idx = max(range(len(results)), key=lambda i: results[i][1])
        self.best_threshold = results[best_idx][0]
        best_fbeta = results[best_idx][1]
        
        print(f"best_threshold: {self.best_threshold:.2f}, fbeta: {best_fbeta:.4f}")
        
        results_df = pd.DataFrame(results, columns=['threshold', 'fbeta', 'precision', 'recall'])
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(results_df['threshold'], results_df['fbeta'], 'b-', linewidth=2, label=f'F{beta}')
        axes[0].plot(results_df['threshold'], results_df['precision'], 'g--', label='Precision')
        axes[0].plot(results_df['threshold'], results_df['recall'], 'r--', label='Recall')
        axes[0].axvline(self.best_threshold, color='orange', linestyle=':', linewidth=2, label=f'Best={self.best_threshold:.2f}')
        axes[0].set_xlabel('Threshold')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Threshold Optimization')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(results_df['recall'], results_df['precision'], 'b-', linewidth=2)
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title('Precision-Recall Curve')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/threshold_optimization.png', dpi=300, bbox_inches='tight')
        print("saved: outputs/threshold_optimization.png")
        
        return self.best_threshold
    
    def evaluate(self, X_test, y_test):
        print("evaluating...")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.best_threshold).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'f2': fbeta_score(y_test, y_pred, beta=2),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        print(f"roc_auc: {metrics['roc_auc']:.4f}")
        print(f"f1: {metrics['f1']:.4f}, f2: {metrics['f2']:.4f}")
        print(f"precision: {metrics['precision']:.4f}, recall: {metrics['recall']:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Clean', 'Fraud']))
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"tp: {tp}, fn: {fn}, fp: {fp}, tn: {tn}")
        print(f"fraud_catch_rate: {tp / (tp + fn) if (tp + fn) > 0 else 0:.2%}")
        print(f"false_positive_rate: {fp / (fp + tn) if (fp + tn) > 0 else 0:.2%}")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Clean', 'Fraud'],
                   yticklabels=['Clean', 'Fraud'])
        plt.title('Confusion Matrix - CatBoost', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('outputs/confusion_matrix_catboost.png', dpi=300, bbox_inches='tight')
        print("saved: outputs/confusion_matrix_catboost.png")
        
        metrics.update({
            'predictions': y_pred,
            'probabilities': y_pred_proba
        })
        
        return metrics
    
    def get_feature_importance(self, top_n=20):
        print(f"feature importance top {top_n}...")
        
        importance = self.model.get_feature_importance()
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        print(feature_importance.head(top_n))
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(top_n)
        plt.barh(range(len(top_features)), top_features['importance'], color='teal')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance - CatBoost', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('outputs/feature_importance_catboost.png', dpi=300, bbox_inches='tight')
        print("saved: outputs/feature_importance_catboost.png")
        
        return feature_importance
    
    def save_model(self, path='models/catboost_model.cbm'):
        self.model.save_model(path)
        print(f"saved: {path}")
    
    def run_full_pipeline(self, data):
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        self.train_model(X_train_sub, y_train_sub, X_val, y_val)
        self.optimize_threshold(X_val, y_val, beta=2)
        metrics = self.evaluate(X_test, y_test)
        feature_importance = self.get_feature_importance(top_n=20)
        self.save_model()
        
        print("done")
        return metrics, feature_importance, (X_test, y_test)


if __name__ == "__main__":
    data = pd.read_csv('data/processed_features.csv')
    
    with open('data/feature_list.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    detector = CatBoostFraudDetector(features)
    metrics, importance, test_data = detector.run_full_pipeline(data)
    
    pd.DataFrame([metrics]).to_csv('outputs/catboost_metrics.csv', index=False)
    importance.to_csv('outputs/catboost_feature_importance.csv', index=False)
    
    print("saved: outputs/")


