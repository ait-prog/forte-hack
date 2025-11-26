import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class FraudDataAnalyzer:
    def __init__(self, trans_path, behavior_path):
        print("downloading...")
        self.trans = pd.read_csv(trans_path, sep=';', encoding='cp1251', header=1)
        self.behavior = pd.read_csv(behavior_path, sep=';', encoding='cp1251', header=1)
        print("ready")
        
    def analyze_transactions(self):
        df = self.trans
        print(f"shape: {df.shape}")
        print(f"columns: {len(df.columns)}")
        
        missing = df.isnull().sum()
        print(f"missing: {missing.sum()}")
        if missing.sum() > 0:
            print(missing[missing > 0])
        
        fraud_counts = df['target'].value_counts()
        fraud_rate = df['target'].mean()
        print(f"target: clean={fraud_counts[0]:,} ({((1-fraud_rate)*100):.2f}%), fraud={fraud_counts[1]:,} ({fraud_rate*100:.2f}%)")
        print(f"imbalance: 1:{int(1/fraud_rate)}")
        
        print(df['amount'].describe())
        amount_by_target = df.groupby('target')['amount'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
        print(amount_by_target)
        
        df['transdate_clean'] = df['transdate'].str.replace("'", "")
        df['transdatetime_clean'] = df['transdatetime'].str.replace("'", "")
        df['trans_date'] = pd.to_datetime(df['transdate_clean'])
        df['trans_datetime'] = pd.to_datetime(df['transdatetime_clean'])
        
        print(f"period: {df['trans_date'].min()} - {df['trans_date'].max()}")
        print(f"days: {(df['trans_date'].max() - df['trans_date'].min()).days}")
        print(f"clients: {df['cst_dim_id'].nunique():,}")
        print(f"avg_trans_per_client: {len(df) / df['cst_dim_id'].nunique():.2f}")
        print(f"fraud_clients: {df[df['target'] == 1]['cst_dim_id'].nunique():,}")
        print(f"recipients: {df['direction'].nunique():,}")
        
        return df
    
    def analyze_behavior(self):
        df = self.behavior
        print(f"shape: {df.shape}, features: {len(df.columns)}")
        
        missing = df.isnull().sum()
        print(f"missing: {missing.sum()}")
        if missing.sum() > 0:
            print(missing[missing > 0])
        
        print(f"phone_models: {df['last_phone_model_categorical'].nunique()}")
        print(df['last_phone_model_categorical'].value_counts().head())
        print(f"os_versions: {df['last_os_categorical'].nunique()}")
        print(df['last_os_categorical'].value_counts().head())
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(f"numeric_features: {len(numeric_cols)}")
        print(f"logins_7d_avg: {df['logins_last_7_days'].mean():.2f}")
        print(f"logins_30d_avg: {df['logins_last_30_days'].mean():.2f}")
        print(f"logins_30d_max: {df['logins_last_30_days'].max()}")
        
        return df
    
    def merge_and_analyze(self):
        self.trans['transdate_clean'] = self.trans['transdate'].str.replace("'", "")
        self.behavior['transdate_clean'] = self.behavior['transdate'].str.replace("'", "")
        
        merged = self.trans.merge(
            self.behavior,
            on=['cst_dim_id', 'transdate_clean'],
            how='left'
        )
        
        print(f"merged_shape: {merged.shape}")
        print(f"with_behavior: {merged['logins_last_7_days'].notna().sum():,}")
        print(f"without_behavior: {merged['logins_last_7_days'].isna().sum():,}")
        
        if merged['logins_last_7_days'].isna().sum() > 0:
            print(merged[merged['logins_last_7_days'].isna()]['target'].value_counts())
        
        return merged
    
    def generate_visualizations(self, merged_df):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        ax = axes[0, 0]
        merged_df['target'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red'])
        ax.set_title('Target Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticklabels(['Clean', 'Fraud'], rotation=0)
        
        ax = axes[0, 1]
        merged_df[merged_df['target'] == 0]['amount'].hist(bins=50, ax=ax, alpha=0.5, 
                                                             label='Clean', color='green', edgecolor='black')
        merged_df[merged_df['target'] == 1]['amount'].hist(bins=50, ax=ax, alpha=0.5, 
                                                             label='Fraud', color='red', edgecolor='black')
        ax.set_title('Amount Distribution by Class', fontsize=14, fontweight='bold')
        ax.set_xlabel('Amount')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_xlim(0, 200000)
        
        ax = axes[0, 2]
        data_clean = merged_df[merged_df['target'] == 0]['logins_last_7_days'].dropna()
        data_fraud = merged_df[merged_df['target'] == 1]['logins_last_7_days'].dropna()
        ax.boxplot([data_clean, data_fraud], labels=['Clean', 'Fraud'])
        ax.set_title('Logins Last 7 Days', fontsize=14, fontweight='bold')
        ax.set_ylabel('Logins Count')
        
        ax = axes[1, 0]
        data_clean = merged_df[merged_df['target'] == 0]['logins_last_30_days'].dropna()
        data_fraud = merged_df[merged_df['target'] == 1]['logins_last_30_days'].dropna()
        ax.boxplot([data_clean, data_fraud], labels=['Clean', 'Fraud'])
        ax.set_title('Logins Last 30 Days', fontsize=14, fontweight='bold')
        ax.set_ylabel('Logins Count')
        
        ax = axes[1, 1]
        data_clean = merged_df[merged_df['target'] == 0]['monthly_phone_model_changes'].dropna()
        data_fraud = merged_df[merged_df['target'] == 1]['monthly_phone_model_changes'].dropna()
        ax.hist([data_clean, data_fraud], bins=10, label=['Clean', 'Fraud'], 
                color=['green', 'red'], alpha=0.6, edgecolor='black')
        ax.set_title('Phone Model Changes (Monthly)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Changes Count')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        ax = axes[1, 2]
        numeric_cols = merged_df.select_dtypes(include=[np.number]).columns
        correlations = merged_df[numeric_cols].corrwith(merged_df['target']).abs().sort_values(ascending=False)[:10]
        correlations.plot(kind='barh', ax=ax, color='teal')
        ax.set_title('Top 10 Features Correlation with Target', fontsize=14, fontweight='bold')
        ax.set_xlabel('|Correlation|')
        
        plt.tight_layout()
        plt.savefig('outputs/eda_visualizations.png', dpi=300, bbox_inches='tight')
        print("saved: outputs/eda_visualizations.png")
        
        return fig
    
    def run_full_analysis(self):
        trans_df = self.analyze_transactions()
        behavior_df = self.analyze_behavior()
        merged_df = self.merge_and_analyze()
        self.generate_visualizations(merged_df)
        print("done")
        return merged_df


if __name__ == "__main__":
    analyzer = FraudDataAnalyzer(
        trans_path='data/transactions.csv',
        behavior_path='data/patterns.csv'
    )
    
    merged_data = analyzer.run_full_analysis()
    merged_data.to_csv('data/merged_data.csv', index=False)
    print("saved: data/merged_data.csv")
