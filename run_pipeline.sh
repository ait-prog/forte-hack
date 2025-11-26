#!/bin/bash
set -e

echo "=== EDA ANALYSIS ==="
python -c "
import importlib.util
import pandas as pd

spec1 = importlib.util.spec_from_file_location('eda', 'src/01_eda_analysis.py')
eda_mod = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(eda_mod)

analyzer = eda_mod.FraudDataAnalyzer('data/transactions.csv', 'data/patterns.csv')
merged = analyzer.run_full_analysis()
merged.to_csv('data/merged_data.csv', index=False)
print('EDA done')
"

echo ""
echo "=== FEATURE ENGINEERING ==="
python -c "
import importlib.util
import pandas as pd

spec2 = importlib.util.spec_from_file_location('fe', 'src/02_feature_engineering.py')
fe_mod = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(fe_mod)

data = pd.read_csv('data/merged_data.csv')
engineer = fe_mod.FraudFeatureEngineer(data)
processed, features = engineer.run_feature_engineering()
processed.to_csv('data/processed_features.csv', index=False)
print('Feature engineering done')
"

echo ""
echo "=== DONE ==="
