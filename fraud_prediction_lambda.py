import json
import os
import pandas as pd
import joblib


MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_detection_pipeline_smote.joblib")

pipeline = joblib.load(MODEL_PATH)

def prepare_data_for_model(df):
    """
    Prepares the input DataFrame for the model by encoding categorical variables and extracting time-based features.

    Args:
        df (pd.DataFrame): The input DataFrame containing transaction data.

    Returns:
        pd.DataFrame: A DataFrame with encoded features ready for model prediction.
    """
    df_encoded = pd.get_dummies(
        df, columns=['device_type', 'location', 'card_type'])
    df_encoded['hour_of_day'] = pd.to_datetime(df_encoded['timestamp']).dt.hour
    df_encoded['day_of_week'] = pd.to_datetime(
        df_encoded['timestamp']).dt.dayofweek
    features = [
        'amount', 'is_vpn', 'hour_of_day', 'day_of_week',
        *[col for col in df_encoded.columns if col.startswith(('device_type_', 'location_', 'card_type_'))]
    ]
    x_encoded = df_encoded[features]
    return x_encoded

def predict_fraud(transaction_json):
    """
    Predicts whether a transaction is fraudulent based on the input transaction data.

    Args:
        transaction_json (str or dict): The transaction data in JSON format (as a string or dictionary).

    Returns:
        dict: A dictionary containing the fraud prediction and the probability of fraud.
    """
    if isinstance(transaction_json, str):
        transaction_data = json.loads(transaction_json)
    elif isinstance(transaction_json, dict):
        transaction_data = transaction_json
    else:
        raise ValueError(
            "Invalid transaction data format. Expected string or dictionary.")
    with open(os.path.join(MODEL_DIR, 'feature_names.json'), 'r', encoding='utf-8') as f:
        feature_names = json.load(f)
    transaction_df = pd.DataFrame([transaction_data])
    if 'status' in transaction_df.columns:
        transaction_df = transaction_df.drop(columns=['status'])
    x_encoded = prepare_data_for_model(transaction_df)
    x_encoded = x_encoded.reindex(columns=feature_names, fill_value=0)
    prediction = pipeline.predict(x_encoded)
    prediction_proba = pipeline.predict_proba(x_encoded)[:, 1]

    return {
        'is_fraud': int(prediction[0]),
        'fraud_probability': float(prediction_proba[0])
    }

def lambda_handler(event, context):
    """
    AWS Lambda handler function that processes incoming events and returns fraud prediction results.

    Args:
        event (dict): The event data passed to the Lambda function.
        context (object): The runtime information provided by AWS Lambda.

    Returns:
        dict: A dictionary containing the fraud prediction and the probability of fraud.
    """
    try:
        # Extract the transaction data from the event
        transaction_json = event.get('transaction_data', {})

        # Predict fraud
        result = predict_fraud(transaction_json)

        # Return the result
        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

# Example usage (for local testing)
if __name__ == "__main__":
    transaction_json = {
        "transaction_id": "T12345",
        "user_id": "U56789",
        "timestamp": "2025-01-01T12:00:00Z",
        "amount": 254.67,
        "device_type": "mobile",
        "location": "California, USA",
        "is_vpn": False,
        "card_type": "credit",
        "status": "approved"
    }
    result = predict_fraud(transaction_json)
    print(result)
