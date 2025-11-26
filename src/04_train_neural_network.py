import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, f1_score, precision_score, 
                             recall_score, fbeta_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("gpu ready")

class NeuralNetworkFraudDetector:
    def __init__(self, features, target='target'):
        self.features = features
        self.target = target
        self.model = None
        self.scaler = StandardScaler()
        self.best_threshold = 0.5
        
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
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"train: {X_train_scaled.shape}, test: {X_test_scaled.shape}")
        print(f"scaled_range: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def build_model(self, input_dim):
        print("building model...")
        
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        model.summary()
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val):
        print("training nn...")
        
        self.model = self.build_model(input_dim=X_train.shape[1])
        
        fraud_weight = len(y_train) / (2 * y_train.sum())
        clean_weight = len(y_train) / (2 * (len(y_train) - y_train.sum()))
        class_weights = {0: clean_weight, 1: fraud_weight}
        
        print(f"class_weights: clean={clean_weight:.2f}, fraud={fraud_weight:.2f}")
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            mode='max'
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=128,
            class_weight=class_weights,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        print(f"trained: {len(history.history['loss'])} epochs")
        self._plot_training_history(history)
        
        return self.model, history
    
    def _plot_training_history(self, history):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(history.history['auc'], label='Train AUC')
        axes[0, 1].plot(history.history['val_auc'], label='Val AUC')
        axes[0, 1].set_title('Model AUC', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/nn_training_history.png', dpi=300, bbox_inches='tight')
        print("saved: outputs/nn_training_history.png")
    
    def optimize_threshold(self, X_val, y_val, beta=2):
        print("optimizing threshold...")
        
        y_pred_proba = self.model.predict(X_val, verbose=0).flatten()
        thresholds = np.arange(0.1, 0.9, 0.05)
        
        best_fbeta = max([(t, fbeta_score(y_val, (y_pred_proba >= t).astype(int), beta=beta))
                         for t in thresholds], key=lambda x: x[1])
        
        self.best_threshold = best_fbeta[0]
        print(f"best_threshold: {self.best_threshold:.2f}, fbeta: {best_fbeta[1]:.4f}")
        
        return self.best_threshold
    
    def evaluate(self, X_test, y_test):
        print("evaluating...")
        
        y_pred_proba = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba >= self.best_threshold).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'f2': fbeta_score(y_test, y_pred, beta=2),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        print(f"roc_auc: {metrics['roc_auc']:.4f}")
        print(f"f1: {metrics['f1']:.4f}, f2: {metrics['f2']:.4f}")
        print(f"precision: {metrics['precision']:.4f}, recall: {metrics['recall']:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Clean', 'Fraud']))
        
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=['Clean', 'Fraud'],
                   yticklabels=['Clean', 'Fraud'])
        plt.title('Confusion Matrix - Neural Network', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('outputs/confusion_matrix_nn.png', dpi=300, bbox_inches='tight')
        print("saved: outputs/confusion_matrix_nn.png")
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {metrics["roc_auc"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Neural Network', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/roc_curve_nn.png', dpi=300, bbox_inches='tight')
        print("saved: outputs/roc_curve_nn.png")
        
        metrics.update({
            'predictions': y_pred,
            'probabilities': y_pred_proba
        })
        
        return metrics
    
    def save_model(self, path='models/nn_model.h5'):
        self.model.save(path)
        joblib.dump(self.scaler, 'models/nn_scaler.pkl')
        print(f"saved: {path}, models/nn_scaler.pkl")
    
    def run_full_pipeline(self, data):
        X_train, X_test, y_train, y_test = self.prepare_data(data)
        
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        self.train_model(X_train_sub, y_train_sub, X_val, y_val)
        self.optimize_threshold(X_val, y_val, beta=2)
        metrics = self.evaluate(X_test, y_test)
        self.save_model()
        
        print("done")
        return metrics, (X_test, y_test)


if __name__ == "__main__":
    data = pd.read_csv('data/processed_features.csv')
    
    with open('data/feature_list.txt', 'r') as f:
        features = [line.strip() for line in f.readlines()]
    
    detector = NeuralNetworkFraudDetector(features)
    metrics, test_data = detector.run_full_pipeline(data)
    
    pd.DataFrame([metrics]).to_csv('outputs/nn_metrics.csv', index=False)
    print("saved: outputs/")

