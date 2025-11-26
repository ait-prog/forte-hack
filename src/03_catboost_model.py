import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings     
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier, Pool

class CatBoostModel:
    def __init__(self, params=None):
        default_params = {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': 42,
            'verbose': 100
        }
        if params:
            default_params.update(params)
        self.params = default_params
        self.model = None

    def train(self, X_train, y_train, X_valid, y_valid):
        train_pool = Pool(X_train, y_train)
        valid_pool = Pool(X_valid, y_valid)

        self.model = CatBoostClassifier(**self.params)
        self.model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
        self.best_iteration_ = self.model.tree_count_

    def predict(self, X):
        if self.model is None:
            raise ValueError("Model is not trained yet.")
        return self.model.predict_proba(X)[:, 1]