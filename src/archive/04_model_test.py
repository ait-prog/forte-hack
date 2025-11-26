from sklearn.model_selection import train_test_split
from catboostmodel import CatBoostModel
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings     
warnings.filterwarnings('ignore')

data = pd.read_csv('../data/processed_features.csv')

X = data.drop(columns=['target'])
y = data['target']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = CatBoostModel()
model.train(X_train, y_train, X_valid, y_valid)
y_pred = model.predict(X_valid)
auc = roc_auc_score(y_valid, y_pred)
accuracy = accuracy_score(y_valid, (y_pred > 0.5).astype(int))

print(f"Validation AUC: {auc:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

feature_importances = model.model.get_feature_importance()
