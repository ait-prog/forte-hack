import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

shap.initjs()

class EnsembleSHAPAnalyzer:
    def __init__(self):
        print("loading models...")
        self.load_models()
        print("ready")
    
    def load_models(self):
        from catboost import CatBoostClassifier
        from tensorflow import keras
        import lightgbm as lgb
        import xgboost as xgb
        from catboost import CatBoostClassifier as CB
        
        self.catboost_model = CatBoostClassifier()
        self.catboost_model.load_model('models/catboost_model.cbm')
        
        self.nn_model = keras.models.load_model('models/nn_model.h5')
        self.scaler = joblib.load('models/nn_scaler.pkl')
        
        prod_config = joblib.load('models/prod_models/config.pkl')
        self.prod_weights = prod_config['weights']
        
        self.prod_models = {}
        self.prod_models['lightgbm'] = lgb.Booster(model_file='models/prod_models/lightgbm_model.txt')
        self.prod_models['xgboost'] = xgb.XGBClassifier()
        self.prod_models['xgboost'].load_model('models/prod_models/xgboost_model.json')
        self.prod_models['catboost'] = CB()
        self.prod_models['catboost'].load_model('models/prod_models/catboost_model.cbm')
        
        try:
            self.prod_feature_selector = joblib.load('models/prod_models/feature_selector.pkl')
            self.prod_selected_features = prod_config['features']
        except:
            self.prod_feature_selector = None
            self.prod_selected_features = None
        
        ensemble_config = joblib.load('models/final_ensemble_config.pkl')
        self.ensemble_weights = ensemble_config['weights']
    
    def prepare_data(self, X):
        X_clean = X.copy()
        for col in X_clean.columns:
            if not pd.api.types.is_numeric_dtype(X_clean[col]):
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_clean.fillna(X_clean.median(), inplace=True)
        return X_clean
    
    def calculate_individual_shap(self, X_sample, sample_size=500):
        print(f"calculating individual shap (sample={sample_size})...")
        
        X_clean = self.prepare_data(X_sample)
        if len(X_clean) > sample_size:
            X_clean = X_clean.sample(n=sample_size, random_state=42)
        
        X_prod = X_clean.copy()
        if self.prod_feature_selector and self.prod_selected_features:
            X_prod_selected = self.prod_feature_selector.transform(X_prod)
            X_prod = pd.DataFrame(
                X_prod_selected,
                columns=self.prod_selected_features,
                index=X_prod.index
            )
        
        shap_values = {}
        
        print("  catboost...")
        explainer_cb = shap.TreeExplainer(self.catboost_model)
        shap_values['catboost'] = explainer_cb.shap_values(X_clean)
        
        print("  lightgbm...")
        explainer_lgb = shap.TreeExplainer(self.prod_models['lightgbm'])
        shap_values['lightgbm'] = explainer_lgb.shap_values(X_prod)
        
        print("  xgboost...")
        explainer_xgb = shap.TreeExplainer(self.prod_models['xgboost'])
        shap_values['xgboost'] = explainer_xgb.shap_values(X_prod)
        
        print("  catboost (prod)...")
        explainer_cb_prod = shap.TreeExplainer(self.prod_models['catboost'])
        shap_values['catboost_prod'] = explainer_cb_prod.shap_values(X_prod)
        
        print(f"done: {X_clean.shape}")
        return shap_values, X_clean, X_prod
    
    def calculate_ensemble_shap(self, shap_values_individual, weights=None, X_prod=None):
        print("calculating ensemble shap...")
        
        if weights is None:
            weights = self.ensemble_weights
        
        w_catboost = weights[0]
        w_nn = weights[1]
        w_prod = weights[2]
        
        w_lgb = self.prod_weights[0] * w_prod
        w_xgb = self.prod_weights[1] * w_prod
        w_cb_prod = self.prod_weights[2] * w_prod
        
        if X_prod is not None and len(X_prod.columns) != len(shap_values_individual['catboost'][0]):
            prod_shap_avg = (w_lgb * shap_values_individual['lightgbm'] +
                            w_xgb * shap_values_individual['xgboost'] +
                            w_cb_prod * shap_values_individual['catboost_prod'])
            
            prod_shap_expanded = np.zeros_like(shap_values_individual['catboost'])
            if self.prod_feature_selector:
                selected_indices = self.prod_feature_selector.get_support(indices=True)
                for i, idx in enumerate(selected_indices):
                    if idx < prod_shap_expanded.shape[1]:
                        prod_shap_expanded[:, idx] = prod_shap_avg[:, i]
            
            ensemble_shap = w_catboost * shap_values_individual['catboost'] + prod_shap_expanded
        else:
            ensemble_shap = (w_catboost * shap_values_individual['catboost'] + 
                            w_lgb * shap_values_individual['lightgbm'] +
                            w_xgb * shap_values_individual['xgboost'] +
                            w_cb_prod * shap_values_individual['catboost_prod'])
        
        print("done")
        return ensemble_shap
    
    def plot_ensemble_shap_summary(self, ensemble_shap, X_sample, features, top_n=20):
        print(f"plotting ensemble shap summary (top {top_n})...")
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            ensemble_shap,
            X_sample,
            feature_names=features,
            max_display=top_n,
            show=False
        )
        plt.title(f'Final Ensemble SHAP Summary - Top {top_n}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('outputs/shap_ensemble_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("saved: outputs/shap_ensemble_summary.png")
    
    def plot_ensemble_shap_bar(self, ensemble_shap, X_sample, features, top_n=20):
        print(f"plotting ensemble shap bar (top {top_n})...")
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            ensemble_shap,
            X_sample,
            feature_names=features,
            plot_type="bar",
            max_display=top_n,
            show=False
        )
        plt.title(f'Final Ensemble SHAP Importance - Top {top_n}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('outputs/shap_ensemble_bar.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("saved: outputs/shap_ensemble_bar.png")
    
    def generate_ensemble_shap_report(self, X_test, features, sample_size=500):
        shap_values_individual, X_sample, X_prod = self.calculate_individual_shap(X_test, sample_size)
        ensemble_shap = self.calculate_ensemble_shap(shap_values_individual, X_prod=X_prod)
        
        self.plot_ensemble_shap_summary(ensemble_shap, X_sample, features, top_n=20)
        self.plot_ensemble_shap_bar(ensemble_shap, X_sample, features, top_n=20)
        
        feature_importance = pd.DataFrame({
            'feature': features,
            'shap_importance': np.abs(ensemble_shap).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        print("top 20 features:")
        print(feature_importance.head(20))
        
        feature_importance.to_csv('outputs/shap_ensemble_importance.csv', index=False)
        print("saved: outputs/shap_ensemble_importance.csv")
        
        print("done")
        return feature_importance, ensemble_shap, X_sample


if __name__ == "__main__":
    data = pd.read_csv('data/processed_features.csv')
    
    with open('data/feature_list.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    from sklearn.model_selection import train_test_split
    
    X = data[features]
    y = data['target']
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    analyzer = EnsembleSHAPAnalyzer()
    feature_importance, ensemble_shap, X_sample = analyzer.generate_ensemble_shap_report(
        X_test, features, sample_size=500
    )
    
    print("saved: outputs/")

