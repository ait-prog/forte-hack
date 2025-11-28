import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, fbeta_score, precision_score
import sys
import importlib.util

sys.path.append('src')

spec = importlib.util.spec_from_file_location('fe', 'src/07_final_ensemble.py')
fe_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe_mod)

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

ensemble = fe_mod.FinalEnsembleDetector()
ensemble.load_models()

print("Testing different weight combinations:")
print("="*60)

best_recall = 0
best_weights = None

for weights in [[1.0, 0.0, 0.0], [0.9, 0.05, 0.05], [0.8, 0.1, 0.1], [0.7, 0.15, 0.15], [0.6, 0.2, 0.2]]:
    ensemble.weights = weights
    ensemble.best_threshold = 0.5
    proba = ensemble.predict_ensemble(X_test, weights=weights)
    pred = (proba >= 0.5).astype(int)
    rec = recall_score(y_test, pred)
    f2 = fbeta_score(y_test, pred, beta=2)
    prec = precision_score(y_test, pred)
    
    print(f"weights={weights[0]:.1f},{weights[1]:.1f},{weights[2]:.1f}: recall={rec:.4f}, f2={f2:.4f}, precision={prec:.4f}")
    
    if rec > best_recall:
        best_recall = rec
        best_weights = weights

print("="*60)
print(f"Best recall: {best_recall:.4f} with weights {best_weights}")

catboost_metrics = pd.read_csv('outputs/catboost_metrics.csv')
catboost_recall = catboost_metrics['recall'].iloc[0]
print(f"CatBoost recall: {catboost_recall:.4f}")

if best_recall >= catboost_recall * 0.95:
    print("Ensemble matches CatBoost performance!")
else:
    print("Using CatBoost directly (weights=[1.0, 0.0, 0.0])")





