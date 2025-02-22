
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
import lightgbm as lgb
from imblearn.over_sampling import ADASYN
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.features = None
        
    def engineer_features(self, df):
        """Creates sophisticated features for fraud detection"""
        df_processed = df.copy()
        
        # Transaction-based features
        df_processed['amount_oldbalanceDest_ratio'] = df_processed['amount'] / (df_processed['oldbalanceDest'] + 1)
        df_processed['amount_oldbalanceOrg_ratio'] = df_processed['amount'] / (df_processed['oldbalanceOrg'] + 1)
        df_processed['balance_diff_org'] = df_processed['newbalanceOrig'] - df_processed['oldbalanceOrg']
        df_processed['balance_diff_dest'] = df_processed['newbalanceDest'] - df_processed['oldbalanceDest']
        
        # Amount-based features
        df_processed['zero_balance_orig'] = (df_processed['oldbalanceOrg'] == 0).astype(int)
        df_processed['zero_balance_dest'] = (df_processed['oldbalanceDest'] == 0).astype(int)
        df_processed['transaction_ratio'] = df_processed['amount'] / (df_processed['oldbalanceOrg'] + 1)
        
        # Step-based features
        df_processed['step_scaled'] = df_processed['step'] / df_processed['step'].max()
        
        # Encode categorical variables
        df_processed['type_encoded'] = self.label_encoder.fit_transform(df_processed['type'])
        
        return df_processed

    def prepare_features(self):
        """Define features used in the model"""
        self.features = ['type_encoded', 'amount', 'step_scaled',
                        'oldbalanceOrg', 'newbalanceOrig', 
                        'oldbalanceDest', 'newbalanceDest',
                        'amount_oldbalanceDest_ratio', 'amount_oldbalanceOrg_ratio',
                        'balance_diff_org', 'balance_diff_dest',
                        'zero_balance_orig', 'zero_balance_dest',
                        'transaction_ratio']
        return self.features

    def train(self, df):
        """Train the fraud detection model"""
        # Process data
        df_processed = self.engineer_features(df)
        self.prepare_features()
        
        X = df_processed[self.features]
        y = df_processed['isFraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance
        adasyn = ADASYN(random_state=42)
        X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train_scaled, y_train)
        
        # Define model parameters
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0,
            'scale_pos_weight': 1,
            'n_estimators': 1000
        }
        
        # Train model
        train_data = lgb.Dataset(X_train_balanced, label=y_train_balanced)
        valid_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)
        
        self.model = lgb.train(
            lgb_params,
            train_data,
            valid_sets=[valid_data],
            early_stopping_rounds=50
        )
        
        # Evaluate model
        y_pred = (self.model.predict(X_test_scaled) > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': (y_pred == y_test).mean(),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, self.model.predict(X_test_scaled))
        }
        
        # Perform cross-validation
        cv_scores = cross_val_score(lgb.LGBMClassifier(**lgb_params), 
                                  X_train_scaled, y_train, 
                                  cv=5, scoring='f1')
        
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"\nCross-validation F1 Scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return metrics

    def save_model(self, model_path='models'):
        """Save the model and preprocessors"""
        import os
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        self.model.save_model(f'{model_path}/fraud_detection_model.txt')
        joblib.dump(self.scaler, f'{model_path}/scaler.joblib')
        joblib.dump(self.label_encoder, f'{model_path}/label_encoder.joblib')
        
    def load_model(self, model_path='models'):
        """Load the model and preprocessors"""
        self.model = lgb.Booster(model_file=f'{model_path}/fraud_detection_model.txt')
        self.scaler = joblib.load(f'{model_path}/scaler.joblib')
        self.label_encoder = joblib.load(f'{model_path}/label_encoder.joblib')

if __name__ == '__main__':
    # Load data
    df = pd.read_csv(r'C:\NPCI\PS_20174392719_1491204439457_log.csv')

    
    # Initialize and train model
    fraud_detector = FraudDetectionModel()
    metrics = fraud_detector.train(df)
    
    # Save model
    fraud_detector.save_model()