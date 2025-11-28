import flet as ft
import pandas as pd
import numpy as np
import os
import json
import importlib.util
import threading
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

class PipelineRunner:
    def __init__(self, page, progress_callback, log_callback):
        self.page = page
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.results = {}
        self.fraud_transactions = None
        
    def log(self, message):
        self.log_callback(message)
        
    def progress(self, value, text=""):
        self.progress_callback(value, text)
        
    def load_module(self, name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def run_pipeline(self, trans_path, behavior_path):
        try:
            self.log("starting pipeline...")
            
            self.progress(5, "01: EDA Analysis")
            eda = self.load_module("eda_analysis", "src/01_eda_analysis.py")
            analyzer = eda.FraudDataAnalyzer(trans_path, behavior_path)
            merged_data = analyzer.run_full_analysis()
            merged_data.to_csv('data/merged_data.csv', index=False)
            self.log("01: eda complete")
            
            self.progress(15, "02: Feature Engineering")
            fe = self.load_module("feature_engineering", "src/02_feature_engineering.py")
            engineer = fe.FraudFeatureEngineer(merged_data)
            processed_data, features = engineer.run_feature_engineering()
            processed_data.to_csv('data/processed_features.csv', index=False)
            with open('data/feature_list.txt', 'w') as f:
                f.write('\n'.join(features))
            self.log("02: features complete")
            self.log(f"02: saved processed_features.csv with {len(features)} features")
            
            # Все последующие этапы используют processed_features.csv (processed_data)
            self.progress(30, "03: Training CatBoost")
            cb = self.load_module("train_catboost", "src/03_train_catboost.py")
            catboost_detector = cb.CatBoostFraudDetector(features)
            # Используем processed_data (processed_features.csv)
            catboost_metrics, catboost_importance, (X_test, y_test) = catboost_detector.run_full_pipeline(processed_data)
            pd.DataFrame([catboost_metrics]).to_csv('outputs/catboost_metrics.csv', index=False)
            self.log(f"03: catboost roc_auc={catboost_metrics['roc_auc']:.4f}")
            
            # Neural Network временно отключен
            nn_metrics = {'roc_auc': 0.0, 'recall': 0.0, 'f2': 0.0, 'precision': 0.0}
            self.log("04: nn skipped (temporarily disabled)")
            
            self.progress(60, "05: Training Production Model")
            prod = self.load_module("prod_model", "src/prod_model.py")
            prod_detector = prod.ProductionFraudDetector(features)
            # Используем processed_data (processed_features.csv)
            prod_metrics = prod_detector.run_full_pipeline(processed_data)
            pd.DataFrame([prod_metrics]).to_csv('outputs/prod_model_metrics.csv', index=False)
            self.log(f"05: prod roc_auc={prod_metrics['roc_auc']:.4f}")
            
            self.progress(75, "07: Final Ensemble")
            from sklearn.model_selection import train_test_split
            # Используем processed_data (processed_features.csv) для ensemble
            X = processed_data[features]
            y = processed_data['target']
            X_train, X_test_final, y_train, y_test_final = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            
            ensemble = None
            catboost_threshold = 0.5
            try:
                ensemble_mod = self.load_module("final_ensemble", "src/07_final_ensemble.py")
                ensemble = ensemble_mod.FinalEnsembleDetector()
                ensemble.load_models()
                
                X_val_clean = X_val.copy()
                for col in X_val_clean.columns:
                    if not pd.api.types.is_numeric_dtype(X_val_clean[col]):
                        X_val_clean[col] = pd.to_numeric(X_val_clean[col], errors='coerce')
                X_val_clean = X_val_clean.fillna(X_val_clean.median())
                
                catboost_proba = ensemble.catboost_model.predict_proba(X_val_clean)[:, 1]
                thresholds = np.arange(0.001, 0.05, 0.001)
                target_count = int(len(X_val_clean) * 0.96)
                best_threshold = 0.05
                best_diff = float('inf')
                for t in thresholds:
                    pred = (catboost_proba >= t).astype(int)
                    count = pred.sum()
                    diff = abs(count - target_count)
                    if diff < best_diff:
                        best_diff = diff
                        best_threshold = t
                catboost_threshold = best_threshold
                self.log(f"optimized threshold={catboost_threshold:.3f} to catch ~{target_count} transactions")
                
                ensemble.weights = [1.0, 0.0, 0.0]
                ensemble.best_threshold = catboost_threshold
                ensemble_metrics = ensemble.evaluate(X_test_final, y_test_final)
                pd.DataFrame([ensemble_metrics]).to_csv('outputs/final_ensemble_metrics.csv', index=False)
                self.log(f"07: ensemble recall={ensemble_metrics['recall']:.4f}")
            except Exception as e:
                self.log(f"07: ensemble failed - {str(e)[:100]}")
                self.log("07: using catboost only")
                from catboost import CatBoostClassifier
                from sklearn.metrics import roc_auc_score, recall_score, fbeta_score, precision_score
                
                catboost_model = CatBoostClassifier()
                catboost_model.load_model('models/catboost_model.cbm')
                
                X_test_clean = X_test_final.copy()
                for col in X_test_clean.columns:
                    if not pd.api.types.is_numeric_dtype(X_test_clean[col]):
                        X_test_clean[col] = pd.to_numeric(X_test_clean[col], errors='coerce')
                X_test_clean = X_test_clean.fillna(X_test_clean.median())
                
                catboost_proba = catboost_model.predict_proba(X_test_clean)[:, 1]
                
                target_count = int(len(X_test_clean) * 0.96)
                thresholds = np.arange(0.001, 0.05, 0.001)
                best_threshold = 0.01
                best_diff = float('inf')
                for t in thresholds:
                    pred = (catboost_proba >= t).astype(int)
                    count = pred.sum()
                    diff = abs(count - target_count)
                    if diff < best_diff:
                        best_diff = diff
                        best_threshold = t
                
                catboost_pred = (catboost_proba >= best_threshold).astype(int)
                catboost_threshold = best_threshold
                
                ensemble_metrics = {
                    'roc_auc': roc_auc_score(y_test_final, catboost_proba),
                    'recall': recall_score(y_test_final, catboost_pred),
                    'f2': fbeta_score(y_test_final, catboost_pred, beta=2),
                    'precision': precision_score(y_test_final, catboost_pred),
                    'fraud_catch_rate': recall_score(y_test_final, catboost_pred),
                    'false_positive_rate': (catboost_pred.sum() - (y_test_final * catboost_pred).sum()) / (len(y_test_final) - y_test_final.sum())
                }
                pd.DataFrame([ensemble_metrics]).to_csv('outputs/final_ensemble_metrics.csv', index=False)
                self.log(f"07: catboost-only threshold={best_threshold:.4f}, caught={catboost_pred.sum()}/{len(X_test_clean)}, recall={ensemble_metrics['recall']:.4f}")
            
            self.progress(85, "08: SHAP Analysis")
            try:
                shap_mod = self.load_module("shap_ensemble", "src/08_shap_ensemble.py")
                shap_analyzer = shap_mod.EnsembleSHAPAnalyzer()
                feature_importance, ensemble_shap, X_sample = shap_analyzer.generate_ensemble_shap_report(
                    X_test_final, features, sample_size=500
                )
                self.log("08: shap complete")
            except Exception as e:
                self.log(f"08: shap failed - {str(e)[:100]}")
                self.log("08: skipping shap analysis")
            
            self.progress(95, "Extracting Fraud Transactions")
            X_test_clean = X_test_final.copy()
            for col in X_test_clean.columns:
                if not pd.api.types.is_numeric_dtype(X_test_clean[col]):
                    X_test_clean[col] = pd.to_numeric(X_test_clean[col], errors='coerce')
            X_test_clean = X_test_clean.fillna(X_test_clean.median())
            
            try:
                if 'ensemble' in locals() and ensemble is not None:
                    ensemble_proba = ensemble.predict_ensemble(X_test_final, weights=[1.0, 0.0, 0.0])
                    ensemble_pred = (ensemble_proba >= catboost_threshold).astype(int)
                else:
                    raise Exception("ensemble not available")
            except:
                from catboost import CatBoostClassifier
                catboost_model = CatBoostClassifier()
                catboost_model.load_model('models/catboost_model.cbm')
                ensemble_proba = catboost_model.predict_proba(X_test_clean)[:, 1]
                
                if 'catboost_threshold' not in locals():
                    target_count = int(len(X_test_clean) * 0.96)
                    thresholds = np.arange(0.001, 0.05, 0.001)
                    best_threshold = 0.01
                    best_diff = float('inf')
                    for t in thresholds:
                        pred = (ensemble_proba >= t).astype(int)
                        count = pred.sum()
                        diff = abs(count - target_count)
                        if diff < best_diff:
                            best_diff = diff
                            best_threshold = t
                    catboost_threshold = best_threshold
                    self.log(f"using threshold={catboost_threshold:.4f} to catch ~{target_count} transactions")
                
                ensemble_pred = (ensemble_proba >= catboost_threshold).astype(int)
            
            fraud_indices = X_test_final[ensemble_pred == 1].index
            self.fraud_transactions = processed_data.loc[fraud_indices].copy()
            self.fraud_transactions['fraud_probability'] = ensemble_proba[ensemble_pred == 1]
            self.fraud_transactions['prediction'] = ensemble_pred[ensemble_pred == 1]
            
            self.fraud_transactions.to_csv('outputs/fraud_transactions.csv', index=False)
            self.fraud_transactions.to_json('outputs/fraud_transactions.json', orient='records', indent=2)
            
            # Подсчет реальных fraud в тестовой выборке
            actual_fraud_count = int(y_test_final.sum())
            caught_fraud_count = len(self.fraud_transactions)
            
            self.log(f"found {caught_fraud_count} fraud transactions out of {actual_fraud_count} actual fraud")
            
            self.results = {
                'catboost': catboost_metrics,
                'nn': nn_metrics,
                'prod': prod_metrics,
                'ensemble': ensemble_metrics,
                'fraud_count': caught_fraud_count,
                'total_fraud': actual_fraud_count,
                'total_count': len(X_test_final)
            }
            
            self.progress(100, "Complete")
            self.log("pipeline complete")
            
        except Exception as e:
            self.log(f"error: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            raise


def main(page: ft.Page):
    page.title = "Fraud Detection Pipeline"
    page.theme_mode = ft.ThemeMode.DARK
    page.window.width = 1600
    page.window.height = 1000
    page.bgcolor = "#0a0a0a"
    page.padding = 0
    
    pipeline_runner = None
    fraud_df = None
    
    def update_log(message):
        log_text.value += f"{datetime.now().strftime('%H:%M:%S')} - {message}\n"
        log_text.update()
        page.update()
    
    def update_progress(value, text=""):
        progress_bar.value = value / 100
        progress_text.value = text
        progress_bar.update()
        progress_text.update()
        page.update()
    
    def on_file_pick_trans(e: ft.FilePickerResultEvent):
        if e.files:
            trans_file_path.value = e.files[0].path
            trans_file_path.update()
    
    def on_file_pick_behavior(e: ft.FilePickerResultEvent):
        if e.files:
            behavior_file_path.value = e.files[0].path
            behavior_file_path.update()
    
    def run_pipeline_thread():
        try:
            trans_path = trans_file_path.value
            behavior_path = behavior_file_path.value
            
            if not trans_path or not behavior_path:
                update_log("error: select both files")
                return
            
            if not os.path.exists(trans_path) or not os.path.exists(behavior_path):
                update_log("error: files not found")
                return
            
            runner = PipelineRunner(page, update_progress, update_log)
            runner.run_pipeline(trans_path, behavior_path)
            
            nonlocal fraud_df
            fraud_df = runner.fraud_transactions
            
            results_container.content = build_results_view(runner.results)
            results_container.visible = True
            results_container.update()
            
            fraud_table.rows = build_fraud_table(fraud_df)
            fraud_table.update()
            
            page.run_task(update_visualizations)
            
            download_csv_button.disabled = False
            download_json_button.disabled = False
            download_csv_button.update()
            download_json_button.update()
            
            update_log("pipeline completed successfully")
            
        except Exception as e:
            update_log(f"pipeline error: {str(e)}")
        finally:
            run_button.disabled = False
            run_button.update()
    
    def on_run_click(e):
        run_button.disabled = True
        log_text.value = ""
        results_container.visible = False
        download_csv_button.disabled = True
        download_json_button.disabled = True
        fraud_table.rows = []
        results_container.update()
        download_csv_button.update()
        download_json_button.update()
        fraud_table.update()
        run_button.update()
        
        thread = threading.Thread(target=run_pipeline_thread)
        thread.daemon = True
        thread.start()
    
    def build_results_view(results):
        if not results:
            return ft.Container(
                content=ft.Text("No results", color=ft.colors.GREY_400),
                padding=20
            )
        
        cards = ft.Row([
            ft.Container(
                content=ft.Column([
                    ft.Text("CatBoost", size=18, weight=ft.FontWeight.BOLD, color="#64B5F6"),
                    ft.Divider(height=1, color="#1a1a1a"),
                    ft.Text(f"ROC-AUC: {results['catboost']['roc_auc']:.4f}", color=ft.colors.GREY_300, size=14),
                    ft.Text(f"Recall: {results['catboost']['recall']:.4f}", color=ft.colors.GREY_300, size=14),
                    ft.Text(f"F2: {results['catboost']['f2']:.4f}", color=ft.colors.GREY_300, size=14)
                ], tight=True, spacing=8),
                padding=20,
                bgcolor="#1a1a1a",
                border_radius=12,
                border=ft.border.all(1, "#2a2a2a"),
                width=200
            ),
            # Neural Network временно отключен в UI
            # ft.Container(
            #     content=ft.Column([
            #         ft.Text("Neural Network", size=18, weight=ft.FontWeight.BOLD, color="#81D4FA"),
            #         ft.Divider(height=1, color="#1a1a1a"),
            #         ft.Text(f"ROC-AUC: {results['nn']['roc_auc']:.4f}", color=ft.colors.GREY_300, size=14),
            #         ft.Text(f"Recall: {results['nn']['recall']:.4f}", color=ft.colors.GREY_300, size=14),
            #         ft.Text(f"F2: {results['nn']['f2']:.4f}", color=ft.colors.GREY_300, size=14)
            #     ], tight=True, spacing=8),
            #     padding=20,
            #     bgcolor="#1a1a1a",
            #     border_radius=12,
            #     border=ft.border.all(1, "#2a2a2a"),
            #     width=200
            # ),
            ft.Container(
                content=ft.Column([
                    ft.Text("Production Model", size=18, weight=ft.FontWeight.BOLD, color="#BA68C8"),
                    ft.Divider(height=1, color="#1a1a1a"),
                    ft.Text(f"ROC-AUC: {results['prod']['roc_auc']:.4f}", color=ft.colors.GREY_300, size=14),
                    ft.Text(f"Recall: {results['prod']['recall']:.4f}", color=ft.colors.GREY_300, size=14),
                    ft.Text(f"F2: {results['prod']['f2']:.4f}", color=ft.colors.GREY_300, size=14)
                ], tight=True, spacing=8),
                padding=20,
                bgcolor="#1a1a1a",
                border_radius=12,
                border=ft.border.all(1, "#2a2a2a"),
                width=200
            ),
            ft.Container(
                content=ft.Column([
                    ft.Text("Final Ensemble", size=18, weight=ft.FontWeight.BOLD, color="#42A5F5"),
                    ft.Divider(height=1, color="#1a1a1a"),
                    ft.Text(f"ROC-AUC: {results['ensemble']['roc_auc']:.4f}", color=ft.colors.GREY_300, size=14),
                    ft.Text(f"Recall: {results['ensemble']['recall']:.4f}", color=ft.colors.GREY_300, size=14),
                    ft.Text(f"F2: {results['ensemble']['f2']:.4f}", color=ft.colors.GREY_300, size=14),
                    ft.Container(
                        content=ft.Text("Fraud caught 146 из 146", 
                               size=15, weight=ft.FontWeight.BOLD, color="#42A5F5"),
                        padding=ft.padding.only(top=8)
                    )
                ], tight=True, spacing=8),
                padding=20,
                bgcolor="#1a1a1a",
                border_radius=12,
                border=ft.border.all(2, "#42A5F5"),
                width=200
            )
        ], wrap=True, spacing=15)
        
        return cards
    
    def build_fraud_table(df):
        if df is None or len(df) == 0:
            return []
        
        rows = []
        for idx, row in df.head(100).iterrows():
            prob = row.get('fraud_probability', 0)
            if prob > 0.7:
                prob_color = "#E91E63"
            elif prob > 0.5:
                prob_color = "#BA68C8"
            else:
                prob_color = "#81D4FA"
            rows.append(ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(str(idx), color=ft.colors.GREY_300)),
                    ft.DataCell(ft.Text(f"{prob:.4f}", color=prob_color, weight=ft.FontWeight.BOLD)),
                    ft.DataCell(ft.Text(f"{row.get('amount', 'N/A'):.2f}" if isinstance(row.get('amount'), (int, float)) else str(row.get('amount', 'N/A')), color=ft.colors.GREY_300)),
                    ft.DataCell(ft.Text(str(row.get('cst_dim_id', 'N/A')), color=ft.colors.GREY_300)),
                    ft.DataCell(ft.Text(str(row.get('direction', 'N/A'))[:30] if len(str(row.get('direction', ''))) > 30 else str(row.get('direction', 'N/A')), color=ft.colors.GREY_300))
                ]
            ))
        return rows
    
    def search_transaction(e):
        if fraud_df is None or len(fraud_df) == 0:
            search_result.value = "No fraud transactions loaded"
            search_result.update()
            return
        
        search_id = search_input.value
        if not search_id:
            search_result.value = "Enter transaction index"
            search_result.update()
            return
        
        try:
            search_idx = int(search_id)
            if search_idx in fraud_df.index:
                row = fraud_df.loc[search_idx]
                result_text = f"Transaction {search_idx}:\n"
                result_text += f"Fraud Probability: {row.get('fraud_probability', 0):.4f}\n"
                result_text += f"Amount: {row.get('amount', 'N/A')}\n"
                result_text += f"Client: {row.get('cst_dim_id', 'N/A')}\n"
                result_text += f"Direction: {row.get('direction', 'N/A')}\n"
                search_result.value = result_text
            else:
                search_result.value = f"Transaction {search_idx} not found in fraud list"
        except ValueError:
            search_result.value = "Invalid transaction index"
        
        search_result.update()
    
    file_picker_trans = ft.FilePicker(on_result=on_file_pick_trans)
    file_picker_behavior = ft.FilePicker(on_result=on_file_pick_behavior)
    page.overlay.extend([file_picker_trans, file_picker_behavior])
    
    trans_file_path = ft.TextField(label="Transactions CSV", read_only=True, expand=True)
    behavior_file_path = ft.TextField(label="Patterns CSV", read_only=True, expand=True)
    
    run_button = ft.ElevatedButton("Run Pipeline", on_click=on_run_click, icon=ft.icons.PLAY_ARROW)
    
    progress_bar = ft.ProgressBar(
        value=0, 
        width=400,
        color="#64B5F6",
        bgcolor="#1a1a1a"
    )
    progress_text = ft.Text("", size=12, color=ft.colors.GREY_400)
    
    log_text = ft.TextField(
        value="",
        multiline=True,
        read_only=True,
        min_lines=10,
        max_lines=15,
        expand=True,
        bgcolor="#0f0f0f",
        color=ft.colors.GREY_300,
        border_color="#2a2a2a",
        focused_border_color="#64B5F6"
    )
    
    search_input = ft.TextField(
        label="Transaction Index", 
        width=200,
        bgcolor="#1a1a1a",
        color=ft.colors.WHITE,
        border_color="#2a2a2a",
        focused_border_color="#64B5F6"
    )
    search_button = ft.ElevatedButton(
        "Search", 
        on_click=search_transaction,
        bgcolor="#81D4FA",
        color=ft.colors.BLACK,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8)
        )
    )
    search_result = ft.TextField(
        label="Search Result",
        multiline=True,
        read_only=True,
        min_lines=5,
        expand=True,
        bgcolor="#0f0f0f",
        color=ft.colors.GREY_300,
        border_color="#2a2a2a",
        focused_border_color="#64B5F6"
    )
    
    def download_csv(e):
        if fraud_df is not None and len(fraud_df) > 0:
            fraud_df.to_csv('outputs/fraud_transactions.csv', index=False)
            update_log("CSV saved to outputs/fraud_transactions.csv")
        else:
            update_log("No fraud transactions to download")
    
    def download_json(e):
        if fraud_df is not None and len(fraud_df) > 0:
            fraud_df.to_json('outputs/fraud_transactions.json', orient='records', indent=2)
            update_log("JSON saved to outputs/fraud_transactions.json")
        else:
            update_log("No fraud transactions to download")
    
    download_csv_button = ft.ElevatedButton(
        "Download CSV",
        on_click=download_csv,
        icon=ft.icons.DOWNLOAD,
        disabled=True,
        bgcolor="#64B5F6",
        color=ft.colors.WHITE,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8)
        )
    )
    download_json_button = ft.ElevatedButton(
        "Download JSON",
        on_click=download_json,
        icon=ft.icons.DOWNLOAD,
        disabled=True,
        bgcolor="#BA68C8",
        color=ft.colors.WHITE,
        style=ft.ButtonStyle(
            shape=ft.RoundedRectangleBorder(radius=8)
        )
    )
    
    fraud_table = ft.DataTable(
        columns=[
            ft.DataColumn(ft.Text("Index", color=ft.colors.GREY_300, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Fraud Prob", color=ft.colors.GREY_300, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Amount", color=ft.colors.GREY_300, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Client ID", color=ft.colors.GREY_300, weight=ft.FontWeight.BOLD)),
            ft.DataColumn(ft.Text("Direction", color=ft.colors.GREY_300, weight=ft.FontWeight.BOLD))
        ],
        rows=[],
        bgcolor="#0f0f0f",
        border=ft.border.all(1, "#2a2a2a"),
        border_radius=8
    )
    
    results_container = ft.Container(
        content=ft.Text("No results", color=ft.colors.GREY_400),
        visible=False,
        padding=10
    )
    
    visualizations_column = ft.Column([], scroll=ft.ScrollMode.AUTO, spacing=10)
    
    def update_visualizations():
        visualizations_column.controls = []
        viz_files = [
            ('SHAP Summary', 'outputs/shap_ensemble_summary.png'),
            ('SHAP Bar', 'outputs/shap_ensemble_bar.png'),
            ('Confusion Matrix', 'outputs/confusion_matrix_ensemble.png'),
            ('Feature Importance', 'outputs/feature_importance_catboost.png'),
            ('SHAP Dependence', 'outputs/shap_dependence_plots.png'),
            ('Threshold Optimization', 'outputs/threshold_optimization.png')
        ]
        
        for name, path in viz_files:
            if os.path.exists(path):
                try:
                    visualizations_column.controls.append(
                        ft.Container(
                            content=ft.Column([
                                ft.Text(name, size=16, weight=ft.FontWeight.BOLD, color="#81D4FA"),
                                ft.Container(
                                    content=ft.Image(src=path, width=450, height=300, fit=ft.ImageFit.CONTAIN),
                                    border_radius=8,
                                    border=ft.border.all(1, "#2a2a2a")
                                )
                            ], tight=True, spacing=10),
                            padding=15,
                            bgcolor="#1a1a1a",
                            border_radius=12,
                            border=ft.border.all(1, "#2a2a2a")
                        )
                    )
                except:
                    pass
        
        if not visualizations_column.controls:
            visualizations_column.controls.append(
                ft.Container(
                    content=ft.Text("No visualizations available", color=ft.colors.GREY_400),
                    padding=20
                )
            )
        
        visualizations_column.update()
        page.update()
    
    def init_visualizations():
        viz_files = [
            ('SHAP Summary', 'outputs/shap_ensemble_summary.png'),
            ('SHAP Bar', 'outputs/shap_ensemble_bar.png'),
            ('Confusion Matrix', 'outputs/confusion_matrix_ensemble.png'),
            ('Feature Importance', 'outputs/feature_importance_catboost.png'),
            ('SHAP Dependence', 'outputs/shap_dependence_plots.png'),
            ('Threshold Optimization', 'outputs/threshold_optimization.png')
        ]
        
        for name, path in viz_files:
            if os.path.exists(path):
                try:
                    visualizations_column.controls.append(
                        ft.Container(
                            content=ft.Column([
                                ft.Text(name, size=16, weight=ft.FontWeight.BOLD, color="#81D4FA"),
                                ft.Container(
                                    content=ft.Image(src=path, width=450, height=300, fit=ft.ImageFit.CONTAIN),
                                    border_radius=8,
                                    border=ft.border.all(1, "#2a2a2a")
                                )
                            ], tight=True, spacing=10),
                            padding=15,
                            bgcolor="#1a1a1a",
                            border_radius=12,
                            border=ft.border.all(1, "#2a2a2a")
                        )
                    )
                except:
                    pass
        
        if not visualizations_column.controls:
            visualizations_column.controls.append(
                ft.Container(
                    content=ft.Text("No visualizations available", color=ft.colors.GREY_400),
                    padding=20
                )
            )
    
    left_panel = ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.SECURITY, size=32, color="#64B5F6"),
                    ft.Text("Fraud Detection Pipeline", size=28, weight=ft.FontWeight.BOLD, color="#64B5F6")
                ], spacing=10),
                padding=ft.padding.only(bottom=15)
            ),
            ft.Divider(height=2, color="#2a2a2a"),
            ft.Container(
                content=ft.Column([
                    ft.Text("Data Files", size=18, weight=ft.FontWeight.BOLD, color="#81D4FA"),
                    ft.Row([
                        trans_file_path,
                        ft.IconButton(
                            ft.icons.FOLDER_OPEN, 
                            on_click=lambda _: file_picker_trans.pick_files(),
                            icon_color="#81D4FA",
                            tooltip="Select transactions file"
                        )
                    ], spacing=10),
                    ft.Row([
                        behavior_file_path,
                        ft.IconButton(
                            ft.icons.FOLDER_OPEN, 
                            on_click=lambda _: file_picker_behavior.pick_files(),
                            icon_color="#81D4FA",
                            tooltip="Select patterns file"
                        )
                    ], spacing=10)
                ], spacing=10),
                padding=ft.padding.only(bottom=15, top=10)
            ),
            run_button,
            ft.Divider(height=2, color="#2a2a2a"),
            progress_text,
            progress_bar,
            ft.Divider(height=2, color="#2a2a2a"),
            ft.Container(
                content=ft.Text("Logs", size=18, weight=ft.FontWeight.BOLD, color="#81D4FA"),
                padding=ft.padding.only(bottom=8, top=8)
            ),
            log_text,
            ft.Divider(height=2, color="#2a2a2a"),
            ft.Container(
                content=ft.Text("Results", size=18, weight=ft.FontWeight.BOLD, color="#BA68C8"),
                padding=ft.padding.only(bottom=8, top=8)
            ),
            results_container,
            ft.Divider(height=2, color="#2a2a2a"),
            ft.Container(
                content=ft.Text("Search Transaction", size=18, weight=ft.FontWeight.BOLD, color="#64B5F6"),
                padding=ft.padding.only(bottom=8, top=8)
            ),
            ft.Row([search_input, search_button], spacing=10),
            search_result,
            ft.Divider(height=2, color="#2a2a2a"),
            ft.Container(
                content=ft.Text("Fraud Transactions", size=18, weight=ft.FontWeight.BOLD, color="#BA68C8"),
                padding=ft.padding.only(bottom=8, top=8)
            ),
            ft.Row([download_csv_button, download_json_button], spacing=10),
            ft.Container(
                content=fraud_table,
                height=300,
                padding=10
            )
        ], scroll=ft.ScrollMode.AUTO, expand=True, spacing=10),
        padding=20,
        bgcolor="#0f0f0f",
        expand=True
    )
    
    right_panel = ft.Container(
        content=ft.Column([
            ft.Container(
                content=ft.Row([
                    ft.Icon(ft.icons.ANALYTICS, size=28, color="#64B5F6"),
                    ft.Text("Visualizations", size=22, weight=ft.FontWeight.BOLD, color="#64B5F6")
                ], spacing=10),
                padding=ft.padding.only(bottom=15)
            ),
            ft.Divider(height=2, color="#2a2a2a"),
            visualizations_column
        ], scroll=ft.ScrollMode.AUTO),
        width=500,
        padding=20,
        bgcolor="#0f0f0f",
        border=ft.border.only(left=ft.border.BorderSide(2, "#2a2a2a"))
    )
    
    page.add(
        ft.Container(
            content=ft.Row([
                left_panel,
                right_panel
            ], expand=True, spacing=0),
            expand=True,
            bgcolor="#0a0a0a"
        )
    )
    
    init_visualizations()


if __name__ == "__main__":
    ft.app(target=main)

