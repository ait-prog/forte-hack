import pandas as pd
import numpy as np

data = pd.read_csv('data/processed_features.csv')

print("=== АНАЛИЗ ПРОБЛЕМЫ ===")
print(f"\n1. Target distribution:")
print(data['target'].value_counts())
print(f"Fraud rate: {data['target'].mean():.4f}")

print(f"\n2. Recipient fraud rate (target encoding):")
print(data['recipient_fraud_rate'].describe())
print(f"\nУникальных получателей: {data['direction'].nunique()}")
print(f"Получателей с fraud_rate > 0: {(data['recipient_fraud_rate'] > 0).sum()}")
print(f"Получателей с fraud_rate = 1: {(data['recipient_fraud_rate'] == 1).sum()}")
print(f"Получателей с fraud_rate = 0: {(data['recipient_fraud_rate'] == 0).sum()}")

print(f"\n3. ПРОБЛЕМА - DATA LEAKAGE:")
print("recipient_fraud_rate вычислен на ВСЕМ датасете!")
print("Это означает, что модель видит информацию о тестовых данных")
print(f"\nПример: если получатель в тесте имеет fraud_rate=1,")
print("это значит, что в train у него тоже был fraud!")

print(f"\n4. Корреляция recipient_fraud_rate с target:")
corr = data['recipient_fraud_rate'].corr(data['target'])
print(f"Корреляция: {corr:.4f}")

print(f"\n5. Проверка утечки:")
fraud_recipients = data[data['target'] == 1]['direction'].unique()
for rec in fraud_recipients[:5]:
    rec_data = data[data['direction'] == rec]
    print(f"\nПолучатель {rec}:")
    print(f"  Всего транзакций: {len(rec_data)}")
    print(f"  Fraud транзакций: {rec_data['target'].sum()}")
    print(f"  recipient_fraud_rate: {rec_data['recipient_fraud_rate'].iloc[0]:.4f}")

