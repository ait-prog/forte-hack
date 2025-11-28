import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sys
import os
sys.path.append('src')

print("=== ПОДСЧЕТ FRAUD, ПОЙМАННЫХ ФИНАЛЬНЫМ АНСАМБЛЕМ ===\n")

data = pd.read_csv('data/processed_features.csv')
y = data['target']

print(f"Всего fraud в данных: {y.sum()} из {len(y)} ({y.mean()*100:.2f}%)\n")

with open('data/feature_list.txt', 'r') as f:
    features = [line.strip() for line in f.readlines()]

X = data[features].copy()
for col in X.columns:
    if not pd.api.types.is_numeric_dtype(X[col]):
        X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Test set: {len(y_test)} транзакций")
print(f"Fraud в test set: {y_test.sum()} из {len(y_test)} ({y_test.mean()*100:.2f}%)\n")

try:
    spec = __import__('importlib.util', fromlist=['spec_from_file_location'])
    import importlib.util
    
    spec = importlib.util.spec_from_file_location('final_ensemble', 'src/07_final_ensemble.py')
    fe_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fe_mod)
    
    detector = fe_mod.FinalEnsembleDetector()
    detector.load_models()
    
    ensemble_proba = detector.predict_ensemble(X_test, weights=detector.weights)
    ensemble_pred = (ensemble_proba >= detector.best_threshold).astype(int)
    
    cm = confusion_matrix(y_test, ensemble_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("="*60)
    print("РЕЗУЛЬТАТЫ ФИНАЛЬНОГО АНСАМБЛЯ:")
    print("="*60)
    print(f"\n📊 Confusion Matrix:")
    print(f"   True Negatives (TN):  {tn:4d} - правильно пропущено clean")
    print(f"   False Positives (FP): {fp:4d} - ложно заблокировано clean")
    print(f"   False Negatives (FN): {fn:4d} - пропущено fraud")
    print(f"   True Positives (TP):  {tp:4d} - поймано fraud ✅")
    
    fraud_catch_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n🎯 РЕЗУЛЬТАТЫ:")
    print(f"   Поймано fraud: {tp} из {y_test.sum()} в test set")
    print(f"   Fraud Catch Rate: {fraud_catch_rate:.2%}")
    print(f"   Пропущено fraud: {fn} из {y_test.sum()}")
    
    total_fraud = y.sum()
    test_fraud = y_test.sum()
    
    print(f"\n📈 МАСШТАБИРОВАНИЕ НА ВЕСЬ ДАТАСЕТ:")
    print(f"   Всего fraud в данных: {total_fraud}")
    print(f"   Fraud в test set: {test_fraud} ({test_fraud/total_fraud*100:.1f}%)")
    estimated_caught = int(tp * total_fraud / test_fraud) if test_fraud > 0 else 0
    print(f"   Оценка пойманных fraud: ~{estimated_caught} из {total_fraud}")
    print(f"   Это примерно {estimated_caught/total_fraud*100:.1f}% от всех fraud")
    
    print(f"\n⚠️  ЛОЖНЫЕ СРАБАТЫВАНИЯ:")
    print(f"   Заблокировано clean: {fp} из {tn + fp}")
    print(f"   False Positive Rate: {fp/(tn+fp)*100:.2f}%")
    
except Exception as e:
    print(f"Ошибка: {e}")
    import traceback
    traceback.print_exc()
    
    print("\nПопробуем из сохраненных метрик...")
    try:
        metrics_df = pd.read_csv('outputs/final_ensemble_metrics.csv')
        recall = metrics_df['recall'].iloc[0]
        test_fraud = y_test.sum()
        tp_estimated = int(recall * test_fraud)
        
        print(f"\nИз сохраненных метрик:")
        print(f"Recall: {recall:.2%}")
        print(f"Fraud в test set: {test_fraud}")
        print(f"Оценка пойманных: {tp_estimated} из {test_fraud}")
        
        total_fraud = y.sum()
        estimated_total = int(tp_estimated * total_fraud / test_fraud)
        print(f"\nМасштабирование на весь датасет:")
        print(f"Оценка пойманных fraud: ~{estimated_total} из {total_fraud}")





