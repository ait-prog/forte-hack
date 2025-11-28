import os
import importlib.util
from datetime import datetime

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print(f"start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

eda = load_module("eda_analysis", "src/01_eda_analysis.py")
analyzer = eda.FraudDataAnalyzer(
    trans_path='data/transactions.csv',
    behavior_path='data/patterns.csv'
)

merged_data = analyzer.run_full_analysis()
merged_data.to_csv('data/merged_data.csv', index=False)

fe = load_module("feature_engineering", "src/02_feature_engineering.py")
engineer = fe.FraudFeatureEngineer(merged_data)
processed_data, features = engineer.run_feature_engineering()

processed_data.to_csv('data/processed_features.csv', index=False)

with open('data/feature_list.txt', 'w') as f:
    f.write('\n'.join(features))

cb = load_module("train_catboost", "src/03_train_catboost.py")
catboost_detector = cb.CatBoostFraudDetector(features)
catboost_metrics, catboost_importance, (X_test, y_test) = catboost_detector.run_full_pipeline(processed_data)

import pandas as pd
pd.DataFrame([catboost_metrics]).to_csv('outputs/catboost_metrics.csv', index=False)
catboost_importance.to_csv('outputs/catboost_feature_importance.csv', index=False)

print(f"catboost: roc_auc={catboost_metrics['roc_auc']:.4f}, f2={catboost_metrics['f2']:.4f}")

nn = load_module("train_neural_network", "src/04_train_neural_network.py")
nn_detector = nn.NeuralNetworkFraudDetector(features)
nn_metrics, (X_test_nn, y_test_nn) = nn_detector.run_full_pipeline(processed_data)

pd.DataFrame([nn_metrics]).to_csv('outputs/nn_metrics.csv', index=False)

print(f"nn: roc_auc={nn_metrics['roc_auc']:.4f}, f2={nn_metrics['f2']:.4f}")

ens = load_module("ensemble_model", "src/05_ensemble_model.py")
from sklearn.model_selection import train_test_split

X = processed_data[features]
y = processed_data['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

ensemble = ens.EnsembleFraudDetector()
ensemble.optimize_weights(X_val, y_val, beta=2)
ensemble_metrics = ensemble.evaluate(X_test, y_test)
comparison = ensemble.compare_models(X_test, y_test)

ensemble.save_ensemble()
pd.DataFrame([ensemble_metrics]).to_csv('outputs/ensemble_metrics.csv', index=False)

print(f"ensemble: roc_auc={ensemble_metrics['roc_auc']:.4f}, f2={ensemble_metrics['f2']:.4f}")
print(f"fraud_catch_rate: {ensemble_metrics['fraud_catch_rate']:.2%}")

shap_mod = load_module("evaluation_shap", "src/06_evaluation_shap.py")
shap_analyzer = shap_mod.FraudSHAPAnalyzer(
    model_path='models/catboost_model.cbm',
    features=features
)

shap_importance, shap_values, X_sample = shap_analyzer.generate_shap_report(
    X_test, y_test, sample_size=1000
)

print("\ncomparison:")
print(comparison.to_string(index=False))

print(f"\nbest: ensemble")
print(f"roc_auc: {ensemble_metrics['roc_auc']:.4f}")
print(f"f2: {ensemble_metrics['f2']:.4f}")
print(f"precision: {ensemble_metrics['precision']:.4f}")
print(f"recall: {ensemble_metrics['recall']:.4f}")
print(f"fraud_catch_rate: {ensemble_metrics['fraud_catch_rate']:.2%}")
print(f"false_positive_rate: {ensemble_metrics['false_positive_rate']:.2%}")

print(f"\nend: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
