import pandas as pd
import numpy as np
import shap
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

shap.initjs()

class FraudSHAPAnalyzer:
    def __init__(self, model_path='models/catboost_model.cbm', features=None):
        print("loading model...")
        self.model = CatBoostClassifier()
        self.model.load_model(model_path)
        self.features = features
        print("ready")
    
    def calculate_shap_values(self, X_sample, sample_size=1000):
        print(f"calculating shap (sample={sample_size})...")
        
        X_clean = X_sample.copy()
        for col in X_clean.columns:
            if not pd.api.types.is_numeric_dtype(X_clean[col]):
                X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        X_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_clean.fillna(X_clean.median(), inplace=True)
        
        if len(X_clean) > sample_size:
            X_clean = X_clean.sample(n=sample_size, random_state=42)
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_clean)
        
        print(f"done: {shap_values.shape}")
        return explainer, shap_values, X_clean
    
    def plot_shap_summary(self, shap_values, X_sample, top_n=20):
        print(f"plotting summary (top {top_n})...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_sample,
            feature_names=self.features,
            max_display=top_n,
            show=False
        )
        plt.title(f'SHAP Summary - Top {top_n} Features', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('outputs/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("saved: outputs/shap_summary_plot.png")
    
    def plot_shap_bar(self, shap_values, X_sample, top_n=20):
        print(f"plotting bar (top {top_n})...")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values, 
            X_sample,
            feature_names=self.features,
            plot_type="bar",
            max_display=top_n,
            show=False
        )
        plt.title(f'SHAP Feature Importance - Top {top_n}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig('outputs/shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("saved: outputs/shap_bar_plot.png")
    
    def plot_shap_dependence(self, shap_values, X_sample, feature_names):
        print("plotting dependence...")
        
        n_features = min(6, len(feature_names))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(feature_names[:n_features]):
            feature_idx = self.features.index(feature)
            shap.dependence_plot(
                feature_idx,
                shap_values,
                X_sample,
                feature_names=self.features,
                ax=axes[i],
                show=False
            )
            axes[i].set_title(f'Dependence: {feature}', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/shap_dependence_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("saved: outputs/shap_dependence_plots.png")
    
    def explain_prediction(self, shap_values, X_sample, idx=0):
        print(f"explaining prediction #{idx}...")
        
        plt.figure(figsize=(12, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=shap_values[idx].sum() - shap_values[idx].sum(),
                data=X_sample.iloc[idx].values,
                feature_names=self.features
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall - Prediction #{idx}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'outputs/shap_waterfall_{idx}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"saved: outputs/shap_waterfall_{idx}.png")
    
    def generate_shap_report(self, X_test, y_test, sample_size=1000):
        explainer, shap_values, X_sample = self.calculate_shap_values(X_test, sample_size)
        
        self.plot_shap_summary(shap_values, X_sample, top_n=20)
        self.plot_shap_bar(shap_values, X_sample, top_n=20)
        
        feature_importance = pd.DataFrame({
            'feature': self.features,
            'shap_importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('shap_importance', ascending=False)
        
        print("top 20 features:")
        print(feature_importance.head(20))
        
        feature_importance.to_csv('outputs/shap_feature_importance.csv', index=False)
        print("saved: outputs/shap_feature_importance.csv")
        
        top_features = feature_importance.head(6)['feature'].tolist()
        self.plot_shap_dependence(shap_values, X_sample, top_features)
        
        fraud_indices = y_test[y_test == 1].index
        if len(fraud_indices) > 0:
            fraud_idx = X_sample.index.get_loc(fraud_indices[0]) if fraud_indices[0] in X_sample.index else 0
            self.explain_prediction(shap_values, X_sample, fraud_idx)
        
        clean_indices = y_test[y_test == 0].index
        if len(clean_indices) > 0:
            clean_idx = X_sample.index.get_loc(clean_indices[0]) if clean_indices[0] in X_sample.index else 1
            self.explain_prediction(shap_values, X_sample, clean_idx)
        
        print("done")
        return feature_importance, shap_values, X_sample


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
    
    analyzer = FraudSHAPAnalyzer(
        model_path='models/catboost_model.cbm',
        features=features
    )
    
    feature_importance, shap_values, X_sample = analyzer.generate_shap_report(
        X_test, y_test, sample_size=1000
    )
    
    print("saved: outputs/")

