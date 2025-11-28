import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, fbeta_score, precision_score, roc_auc_score
import sys
import importlib.util
import joblib

sys.path.append('src')

spec = importlib.util.spec_from_file_location('fe', 'src/07_final_ensemble.py')
fe_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe_mod)

spec_cb = importlib.util.spec_from_file_location('cb', 'src/03_train_catboost.py')
cb_mod = importlib.util.module_from_spec(spec_cb)
spec_cb.loader.exec_module(spec_cb)

data = pd.read_csv('data/processed_features.csv')
with open('data/feature_list.txt', 'r') as f:
    features = [line.strip() for line in f.readlines()]

X = data[features].copy()
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.median())

y = data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Loading CatBoost model to get optimal threshold...")
catboost_detector = cb_mod.CatBoostFraudDetector(features)
catboost_detector.model = catboost_detector.model.__class__()
catboost_detector.model.load_model('models/catboost_model.cbm')

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

X_val_clean = X_val.copy()
for col in X_val_clean.columns:
    if not pd.api.types.is_numeric_dtype(X_val_clean[col]):
        X_val_clean[col] = pd.to_numeric(X_val_clean[col], errors='coerce')
X_val_clean = X_val_clean.fillna(X_val_clean.median())

catboost_proba = catboost_detector.model.predict_proba(X_val_clean)[:, 1]
catboost_threshold = catboost_detector.optimize_threshold(X_val, y_val, beta=2)

print(f"\nCatBoost optimal threshold: {catboost_threshold:.2f}")

print("\nTesting ensemble with CatBoost weights and threshold...")
ensemble = fe_mod.FinalEnsembleDetector()
ensemble.load_models()

ensemble.weights = [1.0, 0.0, 0.0]
ensemble.best_threshold = catboost_threshold

X_test_clean = X_test.copy()
for col in X_test_clean.columns:
    if not pd.api.types.is_numeric_dtype(X_test_clean[col]):
        X_test_clean[col] = pd.to_numeric(X_test_clean[col], errors='coerce')
X_test_clean = X_test_clean.fillna(X_test_clean.median())

proba = ensemble.predict_ensemble(X_test, weights=[1.0, 0.0, 0.0])
pred = (proba >= catboost_threshold).astype(int)

rec = recall_score(y_test, pred)
f2 = fbeta_score(y_test, pred, beta=2)
prec = precision_score(y_test, pred)
roc = roc_auc_score(y_test, proba)

print(f"\nResults with CatBoost (weights=[1.0, 0.0, 0.0], threshold={catboost_threshold:.2f}):")
print(f"  Recall: {rec:.4f}")
print(f"  F2-Score: {f2:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  ROC-AUC: {roc:.4f}")

catboost_metrics = pd.read_csv('outputs/catboost_metrics.csv')
print(f"\nOriginal CatBoost metrics:")
print(f"  Recall: {catboost_metrics['recall'].iloc[0]:.4f}")
print(f"  F2-Score: {catboost_metrics['f2'].iloc[0]:.4f}")

if abs(rec - catboost_metrics['recall'].iloc[0]) < 0.01:
    print("\nSUCCESS! Ensemble matches CatBoost performance.")
    
    config = {
        'weights': [1.0, 0.0, 0.0],
        'best_threshold': catboost_threshold
    }
    joblib.dump(config, 'models/final_ensemble_config.pkl')
    
    metrics = {
        'roc_auc': roc,
        'f1': fbeta_score(y_test, pred, beta=1),
        'f2': f2,
        'precision': prec,
        'recall': rec,
        'fraud_catch_rate': rec,
        'false_positive_rate': (pred.sum() - (y_test * pred).sum()) / (len(y_test) - y_test.sum())
    }
    
    pd.DataFrame([metrics]).to_csv('outputs/final_ensemble_metrics.csv', index=False)
    print("Saved updated config and metrics.")
else:
    print("\nWarning: Results don't match. Check threshold or model loading.")





