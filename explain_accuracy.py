import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=== ПОЧЕМУ ACCURACY ВСЕГДА 0.99 ===\n")

data = pd.read_csv('data/processed_features.csv')
y = data['target']

print("1. РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
print(y.value_counts())
print(f"Fraud: {y.sum()} ({y.mean()*100:.2f}%)")
print(f"Clean: {(len(y) - y.sum())} ({(1-y.mean())*100:.2f}%)\n")

print("2. BASELINE МОДЕЛЬ (всегда предсказываем 'clean'):")
baseline_pred = np.zeros(len(y))
baseline_acc = accuracy_score(y, baseline_pred)
print(f"Accuracy = {baseline_acc:.4f}")
print(f"Это просто доля clean транзакций: {(1-y.mean()):.4f}\n")

print("3. ПОЧЕМУ ACCURACY ВВОДИТ В ЗАБЛУЖДЕНИЕ:")
print(f"Если fraud = {y.mean()*100:.2f}%, то:")
print(f"- Baseline (всегда 0): accuracy = {baseline_acc:.4f}")
print(f"- Любая модель с accuracy > {baseline_acc:.4f} выглядит 'хорошей'")
print(f"- Но это НЕ означает, что модель ловит fraud!\n")

print("4. ПРИМЕР:")
print("Модель с accuracy 0.99 может:")
print("- Правильно предсказать все clean (98.74%)")
print("- НЕ поймать НИ ОДНОГО fraud (0%)")
print("= Accuracy = 0.9874, но модель БЕСПОЛЕЗНА для fraud detection!\n")

print("5. ПРАВИЛЬНЫЕ МЕТРИКИ для несбалансированных данных:")
print("- Precision: из заблокированных, сколько действительно fraud")
print("- Recall: сколько fraud мы поймали из всех fraud")
print("- F2-Score: баланс (приоритет на recall - важно поймать fraud)")
print("- ROC-AUC: способность различать классы")
print("- Fraud Catch Rate: бизнес-метрика (TP / (TP + FN))")
print("- False Positive Rate: сколько clean заблокировано (FP / (FP + TN))\n")

print("6. ВЫВОД:")
print("Accuracy 0.99 - это НОРМАЛЬНО для данных с 1-2% fraud.")
print("Важно смотреть на Precision, Recall, F2-Score и Fraud Catch Rate.")
print("Эти метрики показывают реальную способность модели ловить fraud.")





