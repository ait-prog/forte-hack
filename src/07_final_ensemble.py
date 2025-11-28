import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, 
                             recall_score, fbeta_score, classification_report, 
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

class FinalEnsembleDetector:
    def __init__(self):
        print("loading models...")
        self.catboost_model = None
        self.nn_model = None
        self.prod_model = None
        self.scaler = StandardScaler()
        self.best_threshold = 0.5
        self.weights = None
        
    def load_models(self):
        from catboost import CatBoostClassifier
        
        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model('models/catboost_model.cbm')
        
        try:
            from tensorflow import keras
            self.nn_model = keras.models.load_model('models/nn_model.h5')
            self.scaler = joblib.load('models/nn_scaler.pkl')
        except Exception as e:
            print(f"nn model not available: {str(e)[:100]}")
            self.nn_model = None
            self.scaler = None
        
        prod_config = joblib.load('models/prod_models/config.pkl')
        self.prod_weights = prod_config['weights']
        self.prod_threshold = prod_config['best_threshold']
        
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostClassifier as CB
        
        self.prod_models = {}
        self.prod_models['lightgbm'] = lgb.Booster(model_file='models/prod_models/lightgbm_model.txt')
        self.prod_models['xgboost'] = xgb.XGBClassifier()
        self.prod_models['xgboost'].load_model('models/prod_models/xgboost_model.json')
        self.prod_models['catboost'] = CB()
        self.prod_models['catboost'].load_model('models/prod_models/catboost_model.cbm')
        
        if 'feature_selector' in prod_config or True:
            try:
                self.prod_feature_selector = joblib.load('models/prod_models/feature_selector.pkl')
                self.prod_selected_features = prod_config['features']
            except:
                self.prod_feature_selector = None
                self.prod_selected_features = None
        
        print("ready")
    
    def predict_prod_ensemble(self, X):
        X_clean = X.copy()
        for col in X_clean.columns:
            if not pd.api.types.is_numeric_dtype(X_clean[col]):
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_clean.fillna(X_clean.median(), inplace=True)
        
        if self.prod_feature_selector and self.prod_selected_features:
            X_clean_selected = self.prod_feature_selector.transform(X_clean)
            X_clean = pd.DataFrame(
                X_clean_selected,
                columns=self.prod_selected_features,
                index=X_clean.index
            )
        
        predictions = {}
        predictions['lightgbm'] = self.prod_models['lightgbm'].predict(X_clean, num_iteration=self.prod_models['lightgbm'].best_iteration)
        predictions['xgboost'] = self.prod_models['xgboost'].predict_proba(X_clean)[:, 1]
        predictions['catboost'] = self.prod_models['catboost'].predict_proba(X_clean)[:, 1]
        
        ensemble_proba = sum(w * pred for w, pred in zip(self.prod_weights, predictions.values()))
        return ensemble_proba
    
    def predict_ensemble(self, X, method='weighted_average', weights=None):
        X_clean = X.copy()
        for col in X_clean.columns:
            if not pd.api.types.is_numeric_dtype(X_clean[col]):
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        X_clean = X_clean.fillna(X_clean.median())
        
        catboost_proba = self.catboost_model.predict_proba(X_clean)[:, 1]
        
        if self.nn_model is not None and self.scaler is not None:
            X_scaled = self.scaler.transform(X_clean)
            nn_proba = self.nn_model.predict(X_scaled, verbose=0).flatten()
        else:
            nn_proba = catboost_proba
        
        prod_proba = self.predict_prod_ensemble(X_clean)
        
        if method == 'weighted_average':
            weights = weights or self.weights or [0.33, 0.33, 0.34]
            if self.nn_model is None:
                weights = [weights[0] + weights[1], 0.0, weights[2]]
            ensemble_proba = (weights[0] * catboost_proba + 
                            weights[1] * nn_proba + 
                            weights[2] * prod_proba)
        elif method == 'voting':
            catboost_pred = (catboost_proba >= 0.5).astype(int)
            nn_pred = (nn_proba >= 0.5).astype(int)
            prod_pred = (prod_proba >= 0.5).astype(int)
            ensemble_pred = ((catboost_pred + nn_pred + prod_pred) >= 2).astype(int)
            return ensemble_pred
        
        return ensemble_proba
    
    def optimize_weights(self, X_val, y_val, beta=2, prioritize_recall=True):
        print("optimizing weights...")
        
        X_val_clean = X_val.copy()
        for col in X_val_clean.columns:
            if not pd.api.types.is_numeric_dtype(X_val_clean[col]):
                X_val_clean[col] = pd.to_numeric(X_val_clean[col], errors='coerce')
        X_val_clean = X_val_clean.fillna(X_val_clean.median())
        
        catboost_proba = self.catboost_model.predict_proba(X_val_clean)[:, 1]
        
        thresholds = np.arange(0.1, 0.9, 0.05)
        catboost_results = [(t, recall_score(y_val, (catboost_proba >= t).astype(int)),
                           fbeta_score(y_val, (catboost_proba >= t).astype(int), beta=beta))
                          for t in thresholds]
        catboost_best = max(catboost_results, key=lambda x: x[1] * 0.7 + x[2] * 0.3 if prioritize_recall else x[2])
        catboost_threshold = catboost_best[0]
        catboost_recall = catboost_best[1]
        catboost_fbeta = catboost_best[2]
        
        print(f"catboost optimal: threshold={catboost_threshold:.2f}, recall={catboost_recall:.4f}, fbeta={catboost_fbeta:.4f}")
        
        best_score = catboost_recall * 0.7 + catboost_fbeta * 0.3 if prioritize_recall else catboost_fbeta
        best_weights = [1.0, 0.0, 0.0]
        best_threshold = catboost_threshold
        
        for w1 in np.arange(0.5, 1.01, 0.1):
            for w2 in np.arange(0.0, 1.01 - w1, 0.1):
                w3 = 1.0 - w1 - w2
                if w3 < 0:
                    continue
                weights = [w1, w2, w3]
                
                ensemble_proba = self.predict_ensemble(X_val, weights=weights)
                
                for threshold in np.arange(0.2, 0.7, 0.05):
                    ensemble_pred = (ensemble_proba >= threshold).astype(int)
                    recall = recall_score(y_val, ensemble_pred)
                    fbeta = fbeta_score(y_val, ensemble_pred, beta=beta)
                    
                    if prioritize_recall:
                        score = recall * 0.7 + fbeta * 0.3
                    else:
                        score = fbeta
                    
                    if score > best_score:
                        best_score = score
                        best_weights = weights
                        best_threshold = threshold
        
        self.weights = best_weights
        self.best_threshold = best_threshold
        
        ensemble_proba_final = self.predict_ensemble(X_val, weights=best_weights)
        ensemble_pred_final = (ensemble_proba_final >= best_threshold).astype(int)
        final_recall = recall_score(y_val, ensemble_pred_final)
        final_fbeta = fbeta_score(y_val, ensemble_pred_final, beta=beta)
        
        print(f"weights: catboost={best_weights[0]:.2f}, nn={best_weights[1]:.2f}, prod={best_weights[2]:.2f}")
        print(f"threshold: {best_threshold:.2f}, recall: {final_recall:.4f}, fbeta: {final_fbeta:.4f}")
        
        if best_weights[0] >= 0.95:
            print("using mostly catboost - best for recall")
        
        return best_weights, best_threshold
    
    def evaluate(self, X_test, y_test):
        print("evaluating...")
        
        ensemble_proba = self.predict_ensemble(X_test, weights=self.weights)
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
            'false_positive_rate': false_positive_rate
        })
        
        return metrics


if __name__ == "__main__":
    data = pd.read_csv('data/processed_features.csv')
    
    with open('data/feature_list.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    X = data[features]
    y = data['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    ensemble = FinalEnsembleDetector()
    ensemble.load_models()
    
    X_val_clean = X_val.copy()
    for col in X_val_clean.columns:
        if not pd.api.types.is_numeric_dtype(X_val_clean[col]):
            X_val_clean[col] = pd.to_numeric(X_val_clean[col], errors='coerce')
    X_val_clean = X_val_clean.fillna(X_val_clean.median())
    
    catboost_proba = ensemble.catboost_model.predict_proba(X_val_clean)[:, 1]
    thresholds = np.arange(0.1, 0.9, 0.05)
    catboost_results = [(t, recall_score(y_val, (catboost_proba >= t).astype(int)),
                       fbeta_score(y_val, (catboost_proba >= t).astype(int), beta=2))
                      for t in thresholds]
    catboost_best = max(catboost_results, key=lambda x: x[1] * 0.7 + x[2] * 0.3)
    catboost_threshold = catboost_best[0]
    
    print(f"using catboost directly with threshold={catboost_threshold:.2f}")
    ensemble.weights = [1.0, 0.0, 0.0]
    ensemble.best_threshold = catboost_threshold
    
    metrics = ensemble.evaluate(X_test, y_test)
    
    config = {
        'weights': ensemble.weights,
        'best_threshold': ensemble.best_threshold
    }
    joblib.dump(config, 'models/final_ensemble_config.pkl')
    
    pd.DataFrame([metrics]).to_csv('outputs/final_ensemble_metrics.csv', index=False)
    print("saved: models/final_ensemble_config.pkl, outputs/final_ensemble_metrics.csv")
    print("done")

