import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

print("=== ПОДСЧЕТ FRAUD, ПОЙМАННЫХ ФИНАЛЬНЫМ АНСАМБЛЕМ ===\n")

data = pd.read_csv('data/processed_features.csv')
y = data['target']

print(f"Всего fraud в данных: {y.sum()} из {len(y)} ({y.mean()*100:.2f}%)\n")

with open('data/feature_list.txt', 'r') as f:
    features = [line.strip() for line in f.readlines()]

X = data[features]
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
    from src.final_ensemble import FinalEnsembleDetector
    
    detector = FinalEnsembleDetector()
    detector.load_models()
    
    metrics = detector.evaluate(X_test, y_test)
    
    cm = confusion_matrix(y_test, (detector.predict_ensemble(X_test) >= detector.best_threshold).astype(int))
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ФИНАЛЬНОГО АНСАМБЛЯ:")
    print("="*60)
    print(f"\n✅ Поймано fraud (TP): {tp} из {y_test.sum()}")
    print(f"❌ Пропущено fraud (FN): {fn} из {y_test.sum()}")
    print(f"⚠️  Ложных блокировок (FP): {fp}")
    print(f"✅ Правильно пропущено clean (TN): {tn}")
    
    fraud_catch_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\n📊 Fraud Catch Rate: {fraud_catch_rate:.2%}")
    print(f"📊 Это означает: {tp} из {y_test.sum()} fraud транзакций поймано")
    
    total_fraud = y.sum()
    test_fraud = y_test.sum()
    train_fraud = y_train.sum()
    
    print(f"\n📈 Масштабирование на весь датасет:")
    print(f"   Если в test set {test_fraud} fraud, а всего {total_fraud} fraud,")
    print(f"   то модель поймала бы примерно: {int(tp * total_fraud / test_fraud)} из {total_fraud}")
    
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    print("\nПопробуем загрузить из сохраненных метрик...")
    
    try:
        metrics_df = pd.read_csv('outputs/final_ensemble_metrics.csv')
        print("\nСохраненные метрики:")
        print(metrics_df)
    except:
        print("Метрики не найдены. Запустите 07_final_ensemble.py сначала.")





