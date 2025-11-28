import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                             recall_score, fbeta_score, classification_report, 
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

class ProductionFraudDetector:
    def __init__(self, features, target='target', use_smote=True, feature_selection=True):
        self.features = features
        self.target = target
        self.models = {}
        self.scaler = StandardScaler()
        self.best_threshold = 0.5
        self.weights = None
        self.use_smote = use_smote
        self.feature_selection = feature_selection
        self.feature_selector = None
        self.selected_features = None
        
    def prepare_data(self, data, test_size=0.2, random_state=42):
        print("preparing data...")
        X = data[self.features].copy()
        y = data[self.target].copy()
        
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(X.median(), inplace=True)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        if self.feature_selection:
            print("selecting features...")
            k = min(30, len(X_train.columns))
            self.feature_selector = SelectKBest(f_classif, k=k)
            X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
            selected_indices = self.feature_selector.get_support(indices=True)
            self.selected_features = [X_train.columns[i] for i in selected_indices]
            X_train = pd.DataFrame(
                X_train_selected,
                columns=self.selected_features,
                index=X_train.index
            )
            X_test_selected = self.feature_selector.transform(X_test)
            X_test = pd.DataFrame(
                X_test_selected,
                columns=self.selected_features,
                index=X_test.index
            )
            print(f"selected: {len(self.selected_features)} features")
        
        train_indices_original = X_train.index.copy()
        test_indices_original = X_test.index.copy()
        
        if self.use_smote:
            print("applying smote...")
            train_cols = X_train.columns.tolist()
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            X_train = pd.DataFrame(X_train_resampled, columns=train_cols)
            y_train = pd.Series(y_train_resampled) if isinstance(y_train, pd.Series) else y_train_resampled
            print(f"after smote: {X_train.shape}, fraud_rate: {y_train.mean():.4f}")
        
        print(f"train: {X_train.shape}, test: {X_test.shape}")
        print(f"train_fraud_rate: {y_train.mean():.4f}, test_fraud_rate: {y_test.mean():.4f}")
        
        return X_train, X_test, y_train, y_test, train_indices_original, test_indices_original
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        print("training lightgbm...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 50,
            'learning_rate': 0.03,
            'feature_fraction': 0.75,
            'bagging_fraction': 0.75,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbose': -1,
            'random_state': 42
        }
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )
        
        self.models['lightgbm'] = model
        print(f"lightgbm trained: {model.best_iteration} iterations, auc: {model.best_score['valid_0']['auc']:.4f}")
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        print("training xgboost...")
        
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() if not self.use_smote else 1.0
        
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=7,
            learning_rate=0.03,
            n_estimators=2000,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            early_stopping_rounds=100,
            verbose=100
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.models['xgboost'] = model
        print(f"xgboost trained: best_iteration={model.best_iteration}, auc: {model.best_score:.4f}")
        return model
    
    def train_catboost_optimized(self, X_train, y_train, X_val, y_val):
        print("training catboost optimized...")
        
        model = CatBoostClassifier(
            iterations=2000,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            verbose=200,
            early_stopping_rounds=100,
            auto_class_weights='Balanced' if not self.use_smote else None,
            task_type='CPU',
            border_count=128,
            bagging_temperature=0.5,
            min_data_in_leaf=10
        )
        
        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            plot=False
        )
        
        self.models['catboost'] = model
        print(f"catboost trained: best_iteration={model.best_iteration_}, auc: {model.best_score_['validation']['AUC']:.4f}")
        return model
    
    def predict_ensemble(self, X, weights=None):
        X_clean = X.copy()
        
        for col in X_clean.columns:
            if not pd.api.types.is_numeric_dtype(X_clean[col]):
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        X_clean = X_clean.fillna(X_clean.median())
        
        if self.feature_selection and self.feature_selector and self.selected_features:
            X_clean_selected = self.feature_selector.transform(X_clean)
            X_clean = pd.DataFrame(
                X_clean_selected,
                columns=self.selected_features,
                index=X_clean.index
            )
        
        predictions = {}
        
        if 'lightgbm' in self.models:
            predictions['lightgbm'] = self.models['lightgbm'].predict(X_clean, num_iteration=self.models['lightgbm'].best_iteration)
        
        if 'xgboost' in self.models:
            predictions['xgboost'] = self.models['xgboost'].predict_proba(X_clean)[:, 1]
        
        if 'catboost' in self.models:
            predictions['catboost'] = self.models['catboost'].predict_proba(X_clean)[:, 1]
        
        if not predictions:
            raise ValueError("no models trained")
        
        if weights is None:
            weights = self.weights if self.weights else [1.0/len(predictions)] * len(predictions)
        
        ensemble_proba = sum(w * pred for w, pred in zip(weights, predictions.values()))
        
        return ensemble_proba, predictions
    
    def optimize_weights_and_threshold(self, X_val, y_val, beta=2):
        print("optimizing weights and threshold...")
        
        best_fbeta = 0
        best_weights = [1.0/len(self.models)] * len(self.models)
        best_threshold = 0.5
        
        step = 0.05 if len(self.models) == 3 else 0.1
        
        for w1 in np.arange(0.0, 1.01, step):
            for w2 in np.arange(0.0, 1.01 - w1, step):
                if len(self.models) == 2:
                    weights = [w1, 1.0 - w1]
                elif len(self.models) == 3:
                    w3 = 1.0 - w1 - w2
                    if w3 < 0 or w3 > 1:
                        continue
                    weights = [w1, w2, w3]
                else:
                    weights = [1.0/len(self.models)] * len(self.models)
                
                ensemble_proba, _ = self.predict_ensemble(X_val, weights=weights)
                
                for threshold in np.arange(0.2, 0.8, 0.02):
                    ensemble_pred = (ensemble_proba >= threshold).astype(int)
                    fbeta = fbeta_score(y_val, ensemble_pred, beta=beta)
                    
                    if fbeta > best_fbeta:
                        best_fbeta = fbeta
                        best_weights = weights
                        best_threshold = threshold
        
        self.weights = best_weights
        self.best_threshold = best_threshold
        
        model_names = list(self.models.keys())
        weight_str = ', '.join([f"{name}={w:.2f}" for name, w in zip(model_names, best_weights)])
        print(f"weights: {weight_str}")
        print(f"threshold: {best_threshold:.2f}, fbeta: {best_fbeta:.4f}")
        
        return best_weights, best_threshold
    
    def evaluate(self, X_test, y_test):
        print("evaluating...")
        
        ensemble_proba, _ = self.predict_ensemble(X_test, weights=self.weights)
        ensemble_pred = (ensemble_proba >= self.best_threshold).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, ensemble_proba),
            'f1': f1_score(y_test, ensemble_pred),
            'f2': fbeta_score(y_test, ensemble_pred, beta=2),
            'precision': precision_score(y_test, ensemble_pred),
            'recall': recall_score(y_test, ensemble_pred)
        }
        
        print(f"roc_auc: {metrics['roc_auc']:.4f}")
        print(f"f1: {metrics['f1']:.4f}, f2: {metrics['f2']:.4f}")
        print(f"precision: {metrics['precision']:.4f}, recall: {metrics['recall']:.4f}")
        print(classification_report(y_test, ensemble_pred, target_names=['Clean', 'Fraud']))
        
        cm = confusion_matrix(y_test, ensemble_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"tp: {tp}, fn: {fn}, fp: {fp}, tn: {tn}")
        fraud_catch_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        print(f"fraud_catch_rate: {fraud_catch_rate:.2%}")
        print(f"false_positive_rate: {false_positive_rate:.2%}")
        
        metrics.update({
            'fraud_catch_rate': fraud_catch_rate,
            'false_positive_rate': false_positive_rate,
            'predictions': ensemble_pred,
            'probabilities': ensemble_proba
        })
        
        return metrics
    
    def save_models(self, path='models/prod_models'):
        import os
        os.makedirs(path, exist_ok=True)
        
        if 'lightgbm' in self.models:
            self.models['lightgbm'].save_model(f'{path}/lightgbm_model.txt')
        
        if 'xgboost' in self.models:
            self.models['xgboost'].save_model(f'{path}/xgboost_model.json')
        
        if 'catboost' in self.models:
            self.models['catboost'].save_model(f'{path}/catboost_model.cbm')
        
        config = {
            'weights': self.weights,
            'best_threshold': self.best_threshold,
            'features': self.selected_features if self.selected_features else self.features,
            'use_smote': self.use_smote,
            'feature_selection': self.feature_selection
        }
        joblib.dump(config, f'{path}/config.pkl')
        if self.feature_selector:
            joblib.dump(self.feature_selector, f'{path}/feature_selector.pkl')
        
        print(f"saved: {path}/")
    
    def run_full_pipeline(self, data):
        X_train, X_test, y_train, y_test, train_indices_original, test_indices_original = self.prepare_data(data)
        
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        if self.use_smote:
            val_size = len(X_val)
            val_indices_in_original = train_indices_original[:val_size]
        else:
            val_indices_in_original = X_val.index if hasattr(X_val, 'index') else range(len(X_val))
        
        X_val_original = data[self.features].loc[val_indices_in_original].copy()
        for col in X_val_original.columns:
            if not pd.api.types.is_numeric_dtype(X_val_original[col]):
                X_val_original[col] = pd.to_numeric(X_val_original[col], errors='coerce')
        X_val_original.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_val_original.fillna(X_val_original.median(), inplace=True)
        
        X_test_original = data[self.features].loc[test_indices_original].copy()
        for col in X_test_original.columns:
            if not pd.api.types.is_numeric_dtype(X_test_original[col]):
                X_test_original[col] = pd.to_numeric(X_test_original[col], errors='coerce')
        X_test_original.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_test_original.fillna(X_test_original.median(), inplace=True)
        
        self.train_lightgbm(X_train_sub, y_train_sub, X_val, y_val)
        self.train_xgboost(X_train_sub, y_train_sub, X_val, y_val)
        self.train_catboost_optimized(X_train_sub, y_train_sub, X_val, y_val)
        
        self.optimize_weights_and_threshold(X_val_original, y_val, beta=2)
        metrics = self.evaluate(X_test_original, y_test)
        self.save_models()
        
        print("done")
        return metrics


if __name__ == "__main__":
    data = pd.read_csv('data/processed_features.csv')
    
    with open('data/feature_list.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    detector = ProductionFraudDetector(features, use_smote=True, feature_selection=True)
    metrics = detector.run_full_pipeline(data)
    
    pd.DataFrame([metrics]).to_csv('outputs/prod_model_metrics.csv', index=False)
    print("saved: outputs/prod_model_metrics.csv")



