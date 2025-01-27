import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import seaborn as sns
from imblearn.over_sampling import SMOTE


DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_mock_transactions(num_transactions=2000):
    """
    Generates a mock dataset of transactions with synthetic features and labels.

    Args:
        num_transactions (int): The number of transactions to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the generated transaction data.
    """
    np.random.seed(42)
    device_types = ['mobile', 'desktop', 'tablet']
    locations = ['California, USA', 'New York, USA', 'Texas, USA', 'Florida, USA']
    card_types = ['credit', 'debit', 'prepaid']

    transactions = []
    base_time = pd.Timestamp.now()

    for _ in range(num_transactions):

        fraud_prob = np.random.choice([0, 1], p=[0.7, 0.3])

        transaction = {
            "transaction_id": f"T{np.random.randint(10000, 99999)}",
            "user_id": f"U{np.random.randint(10000, 99999)}",
            "timestamp": (base_time + pd.Timedelta(days=np.random.randint(0, 365))).isoformat() + "Z",
            "amount": round(np.random.exponential(scale=fraud_prob * 500 + 250), 2),
            "device_type": np.random.choice(device_types),
            "location": np.random.choice(locations),
            "is_vpn": np.random.choice([True, False], p=[0.3 if fraud_prob else 0.2, 0.7 if fraud_prob else 0.8]),
            "card_type": np.random.choice(card_types),
            "is_fraud": fraud_prob
        }
        transactions.append(transaction)

    return pd.DataFrame(transactions)

def prepare_data_for_model(df):
    """
    Prepares the input DataFrame for model training by encoding categorical features and extracting time-based features.

    Args:
        df (pd.DataFrame): The input DataFrame containing transaction data.

    Returns:
        tuple: A tuple containing the feature matrix (X) and target vector (y).
    """
    df_encoded = pd.get_dummies(df, columns=['device_type', 'location', 'card_type'])

    df_encoded['hour_of_day'] = pd.to_datetime(df_encoded['timestamp']).dt.hour
    df_encoded['day_of_week'] = pd.to_datetime(df_encoded['timestamp']).dt.dayofweek

    features = [
        'amount', 'is_vpn', 'hour_of_day', 'day_of_week',
        *[col for col in df_encoded.columns if col.startswith(('device_type_', 'location_', 'card_type_'))]
    ]

    X = df_encoded[features]
    y = df_encoded['is_fraud']

    return X, y

def train_and_evaluate_model(X, y):
    """
    Trains and evaluates a fraud detection model using a pipeline with SMOTE for handling class imbalance.

    Args:
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.

    Returns:
        tuple: A tuple containing the trained pipeline, test data, predictions, and prediction probabilities.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train.to_csv(os.path.join(DATA_DIR, "X_train_original.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_DIR, "X_test_original.csv"), index=False)
    y_train.to_csv(os.path.join(DATA_DIR, "y_train_original.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, "y_test_original.csv"), index=False)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    pd.DataFrame(X_train_resampled).to_csv(os.path.join(DATA_DIR, "X_train_resampled.csv"), index=False)
    pd.Series(y_train_resampled).to_csv(os.path.join(DATA_DIR, "y_train_resampled.csv"), index=False)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight="balanced"
            ))
    ])

    cv_scores = cross_val_score(pipeline, X_train_resampled, y_train_resampled, cv=5)
    print("Cross-validation scores:", cv_scores)
    print("Mean CV score:", cv_scores.mean())

    pipeline.fit(X_train_resampled, y_train_resampled)

    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    feature_names = list(X_train_resampled.columns)
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'w', encoding='utf-8') as f:
        json.dump(feature_names, f)

    joblib.dump(pipeline, os.path.join(MODEL_DIR, 'fraud_detection_pipeline_smote.joblib'))

    return pipeline, X_test, y_test, y_pred, y_pred_proba

def plot_model_evaluation(y_test, y_pred, y_pred_proba):
    """
    Plots evaluation metrics for the model, including a confusion matrix and ROC curve.

    Args:
        y_test (pd.Series): The true labels for the test set.
        y_pred (np.ndarray): The predicted labels for the test set.
        y_pred_proba (np.ndarray): The predicted probabilities for the test set.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Fraud', 'Fraud'],
                yticklabels=['Not Fraud', 'Fraud'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'fraud_model_evaluation_smote.png'))
    plt.close()

def main():
    """
    Main function to generate mock transaction data, prepare it for modeling, train and evaluate the model, and visualize results.
    """
    transactions_df = generate_mock_transactions(10000)

    transactions_df.to_csv(os.path.join(DATA_DIR, "original_transactions.csv"), index=False)

    X, y = prepare_data_for_model(transactions_df)

    pipeline, X_test, y_test, y_pred, y_pred_proba = train_and_evaluate_model(X, y)

    plot_model_evaluation(y_test, y_pred, y_pred_proba)

if __name__ == "__main__":
    main()
