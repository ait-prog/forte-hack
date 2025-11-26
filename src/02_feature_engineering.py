import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FraudFeatureEngineer:
    def __init__(self, data):
        self.data = data.copy()
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def create_temporal_features(self):
        print("temporal features...")
        df = self.data
        
        if 'trans_datetime' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['trans_datetime']):
            if 'transdatetime_clean' in df.columns:
                df['trans_datetime'] = pd.to_datetime(df['transdatetime_clean'])
            elif 'transdatetime' in df.columns:
                df['transdatetime_clean'] = df['transdatetime'].str.replace("'", "")
                df['trans_datetime'] = pd.to_datetime(df['transdatetime_clean'])
            else:
                raise ValueError("No datetime column found")
        
        df['hour'] = df['trans_datetime'].dt.hour
        df['day_of_week'] = df['trans_datetime'].dt.dayofweek
        df['day_of_month'] = df['trans_datetime'].dt.day
        df['month'] = df['trans_datetime'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        df['time_of_day'] = pd.cut(df['hour'], 
                                     bins=[0, 6, 12, 18, 24],
                                     labels=['night', 'morning', 'afternoon', 'evening'],
                                     include_lowest=True)
        
        df['is_suspicious_hour'] = ((df['hour'] >= 0) & (df['hour'] <= 5)).astype(int)
        
        print(f"created: 7")
        return df
    
    def create_amount_features(self):
        print("amount features...")
        df = self.data
        
        df['amount_log'] = np.log1p(df['amount'])
        
        df['amount_category'] = pd.cut(df['amount'],
                                        bins=[0, 1000, 5000, 20000, 50000, np.inf],
                                        labels=['very_small', 'small', 'medium', 'large', 'very_large'])
        
        df['is_round_amount'] = (df['amount'] % 10000 == 0).astype(int)
        df['is_round_1000'] = (df['amount'] % 1000 == 0).astype(int)
        
        client_amount_stats = df.groupby('cst_dim_id')['amount'].agg(['mean', 'std', 'min', 'max', 'count']).add_prefix('client_amount_')
        df = df.merge(client_amount_stats, left_on='cst_dim_id', right_index=True, how='left')
        
        df['amount_deviation_from_mean'] = (df['amount'] - df['client_amount_mean']) / (df['client_amount_std'] + 1)
        df['is_unusual_amount'] = (df['amount'] > df['client_amount_mean'] + 2 * df['client_amount_std']).astype(int)
        
        print(f"created: 11")
        return df
    
    def create_client_features(self):
        print("client features...")
        df = self.data
        
        client_trans_count = df.groupby('cst_dim_id').size().rename('client_trans_count')
        df = df.merge(client_trans_count, left_on='cst_dim_id', right_index=True, how='left')
        
        client_unique_recipients = df.groupby('cst_dim_id')['direction'].nunique().rename('client_unique_recipients')
        df = df.merge(client_unique_recipients, left_on='cst_dim_id', right_index=True, how='left')
        
        df['recipient_frequency_ratio'] = 1 / (df['client_unique_recipients'] + 1)
        
        print(f"created: 3")
        return df
    
    def create_recipient_features(self):
        print("recipient features...")
        df = self.data
        
        recipient_popularity = df.groupby('direction').size().rename('recipient_popularity')
        df = df.merge(recipient_popularity, left_on='direction', right_index=True, how='left')
        
        recipient_avg_amount = df.groupby('direction')['amount'].mean().rename('recipient_avg_amount')
        df = df.merge(recipient_avg_amount, left_on='direction', right_index=True, how='left')
        
        df['is_rare_recipient'] = (df['recipient_popularity'] <= 2).astype(int)
        
        print(f"created: 3")
        return df
    
    def create_behavioral_features(self):
        print("behavioral features...")
        df = self.data
        
        if 'logins_last_7_days' not in df.columns:
            print("skipping: no behavioral data")
            return df
        
        numeric_cols = ['freq_change_7d_vs_mean', 'monthly_phone_model_changes', 
                       'monthly_os_changes', 'burstiness_login_interval',
                       'logins_last_7_days', 'logins_last_30_days']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['login_spike'] = (df['freq_change_7d_vs_mean'] > 2).astype(int)
        df['login_drop'] = (df['freq_change_7d_vs_mean'] < -0.5).astype(int)
        df['device_changed_recently'] = (df['monthly_phone_model_changes'] > 1).astype(int)
        df['os_changed_recently'] = (df['monthly_os_changes'] > 1).astype(int)
        df['high_burstiness'] = (df['burstiness_login_interval'] > 0.5).astype(int)
        df['low_activity'] = ((df['logins_last_7_days'] == 0) | (df['logins_last_30_days'] <= 5)).astype(int)
        df['login_to_trans_ratio'] = df['logins_last_7_days'] / (df['client_trans_count'] + 1)
        
        print(f"created: 7")
        return df
    
    def encode_categorical(self):
        print("encoding categorical...")
        df = self.data
        
        categorical_cols = ['last_phone_model_categorical', 'last_os_categorical', 
                           'time_of_day', 'amount_category']
        
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        target_mean = df.groupby('direction')['target'].mean().rename('recipient_fraud_rate')
        df = df.merge(target_mean, left_on='direction', right_index=True, how='left')
        
        print(f"encoded: {len(categorical_cols) + 1}")
        return df
    
    def select_features(self):
        print("selecting features...")
        df = self.data
        
        base_features = ['amount', 'amount_log', 'hour', 'day_of_week', 'day_of_month', 
                        'month', 'is_weekend', 'is_suspicious_hour', 'is_round_amount',
                        'is_round_1000', 'client_amount_mean', 'client_amount_std',
                        'client_amount_min', 'client_amount_max', 'client_amount_count',
                        'amount_deviation_from_mean', 'is_unusual_amount',
                        'client_trans_count', 'client_unique_recipients',
                        'recipient_frequency_ratio', 'recipient_popularity',
                        'recipient_avg_amount', 'is_rare_recipient',
                        'recipient_fraud_rate']
        
        behavioral_features = ['logins_last_7_days', 'logins_last_30_days',
                              'login_frequency_7d', 'login_frequency_30d',
                              'freq_change_7d_vs_mean', 'logins_7d_over_30d_ratio',
                              'avg_login_interval_30d', 'std_login_interval_30d',
                              'monthly_phone_model_changes', 'monthly_os_changes',
                              'burstiness_login_interval', 'login_spike', 'login_drop',
                              'device_changed_recently', 'os_changed_recently',
                              'high_burstiness', 'low_activity', 'login_to_trans_ratio']
        
        encoded_features = [col for col in df.columns if col.endswith('_encoded')]
        
        all_features = base_features + behavioral_features + encoded_features
        available_features = [f for f in all_features if f in df.columns]
        
        print(f"selected: {len(available_features)}")
        return df, available_features
    
    def handle_missing_values(self, feature_columns):
        print("handling missing...")
        df = self.data
        
        fill_func = lambda col: df[col].fillna(df[col].median() if df[col].dtype in ['float64', 'int64'] else df[col].mode()[0], inplace=True)
        [fill_func(col) for col in feature_columns if df[col].isnull().sum() > 0]
        
        print("done")
        return df
    
    def run_feature_engineering(self):
        self.data = self.create_temporal_features()
        self.data = self.create_amount_features()
        self.data = self.create_client_features()
        self.data = self.create_recipient_features()
        self.data = self.create_behavioral_features()
        self.data = self.encode_categorical()
        
        self.data, feature_columns = self.select_features()
        self.data = self.handle_missing_values(feature_columns)
        
        print(f"total_features: {len(feature_columns)}, shape: {self.data.shape}")
        return self.data, feature_columns


if __name__ == "__main__":
    data = pd.read_csv('data/merged_data.csv')
    
    engineer = FraudFeatureEngineer(data)
    processed_data, features = engineer.run_feature_engineering()
    
    processed_data.to_csv('data/processed_features.csv', index=False)
    
    with open('data/feature_list.txt', 'w') as f:
        f.write('\n'.join(features))
    
    print("saved: data/processed_features.csv, data/feature_list.txt")

