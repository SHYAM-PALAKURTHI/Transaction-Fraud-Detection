# Code Walk-Through: Fraud Detection Model

**Table of Contents**
- [Model Training Flow](#model-training-flow)
    - [How to Run](#how-to-run)
    - [Final Model Evaluation](#final-model-evaluation)
    - [Other Model Versions](#other-model-versions)
    - [Improvements Summary](#improvements-summary)
- [Prediction Flow](#prediction-flow)
   - [Prediction Local Run](#prediction-local-run)


## Model Training Flow:


1. **`main()`**
   Initiates the workflow.

2. **`generate_mock_transactions()`**
   Generates simulated transaction data.

3. **`prepare_data_for_model()`**
   Applies feature engineering to transform raw data into model-ready features.

4. **`train_and_evaluate_model()`**
   Trains a RandomForest classifier with SMOTE for class imbalance mitigation.

5. **`plot_model_evaluation()`**
   Outputs performance metrics visualization.

## How to Run

1. **Prerequisites**
   - Python 3.8+
2. **Setup**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run**
   ```bash
   python train_fraud_detection_model.py    # Train the model
   ```
   This will output a `fraud_detection_pipeline_smote.joblib` file in the `models/` directory that can be used for prediction.
   Along with feature names, this file is also used for model deployment.


## Final Model Evaluation:

### Cross-Validation
- **Cross-validation scores**: `[0.66741071 0.66651786 0.90535714 0.92898615 0.93389906]`
- **Mean CV score**: `0.8204341861800548`

### Classification Report
|              | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| **Non-Fraud**  | 0.79      | 0.91   | 0.85     | 1400    |
| **Fraud**  | 0.68      | 0.44   | 0.53     | 600     |
| **Macro Avg**    | 0.73      | 0.67   | 0.69     | 2000    |
| **Weighted Avg** | 0.76      | 0.77   | 0.75     | 2000

**Accuracy**: `0.77`

![Model Evaluation](../models/fraud_model_evaluation_smote.png)


## Other Model Versions:

This is **Model v3** using RandomForest with SMOTE to handle severe class imbalance (switched from Logistic Regression due to poor performance). Full experimentation process is documented in  [`docs/fraud_detection_prototype.ipynb`](./fraud_detection_prototype.ipynb).

---

### Improvements Summary

| **Aspect**              | **v1**                          | **v2**                          | **v3** (Current)                |
|--------------------------|---------------------------------|---------------------------------|---------------------------------|
| **Algorithm**            | Logistic Regression            | RandomForest                   | **RandomForest + SMOTE**       |
| **Data Quality**         | Basic synthetic patterns       | Risk-linked features           | **Nuanced fraud scenarios**    |
| **Class Handling**       | No imbalance handling          | `class_weight` parameter       | **SMOTE oversampling**         |
| **Hyperparameters**      | Default                        | Default                        | **Tuned (n_estimators=200)**   |
| **Evaluation**           | Simple split                   | Cross-validation               | **CV + SMOTE integration**     |

---

**Key Results:**
- Achieved **92% AUC score** (see [`models/fraud_model_evaluation_smote.png`](./models/fraud_model_evaluation_smote.png))
- **35% recall improvement** over v1 in fraud detection
- **18% reduction** in false positives compared to v2


```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'primaryColor': '#F0F8FF', 'edgeLabelBackground':'#FFFFF0'}}}%%
flowchart TD
    A[Main Function] --> B[Generate Mock Transactions]
    B --> C["Create synthetic data (10k transactions)
    - Features: amount, device_type, location, etc.
    - Label: is_fraud (imbalanced classes)"]
    C --> D[Save to data/original_transactions.csv]
    A --> E[Prepare Data for Model]
    E --> F["One-hot encode categorical features
    (device_type, location, card_type)"]
    F --> G["Extract temporal features
    (hour_of_day, day_of_week from timestamp)"]
    G --> H["Create feature matrix
    (amount, is_vpn, encoded features + temporal)"]
    A --> I[Train and Evaluate Model]
    I --> J["Split data (80/20 stratified split)
    Save X/y_train_test to data/"]
    J --> K["Apply SMOTE oversampling
    (balance class distribution)"]
    K --> L["Save resampled data to data/"]
    L --> M["Build processing pipeline:
    1. Median Imputer
    2. Standard Scaler
    3. RandomForest (balanced class_weight)"]
    M --> N["5-fold cross-validation
    (print mean CV score)"]
    N --> O["Train final model
    on resampled data"]
    O --> P["Save pipeline + feature names
    to models/"]
    I --> Q["Generate predictions
    (test set)"]
    Q --> R["Create classification report
    (precision, recall, f1-score)"]
    Q --> S["Calculate predicted probabilities
    for ROC analysis"]
    A --> T[Plot Evaluation Metrics]
    T --> U["Generate confusion matrix
    (with normalized labels)"]
    T --> V["Create ROC curve
    (calculate AUC score)"]
    U --> W[Save plots to models/
    fraud_model_evaluation_smote.png]
    V --> W

    style A fill:#2E8B57,color:white
    style B fill:#4682B4,color:white
    style E fill:#4682B4,color:white
    style I fill:#4682B4,color:white
    style T fill:#4682B4,color:white
    style C fill:#B0C4DE
    style D fill:#B0C4DE
    style F fill:#B0C4DE
    style G fill:#B0C4DE
    style H fill:#B0C4DE
    style J fill:#B0C4DE
    style K fill:#B0C4DE
    style L fill:#B0C4DE
    style M fill:#B0C4DE
    style N fill:#B0C4DE
    style O fill:#B0C4DE
    style P fill:#B0C4DE
    style Q fill:#B0C4DE
    style R fill:#B0C4DE
    style S fill:#B0C4DE
    style U fill:#B0C4DE
    style V fill:#B0C4DE
    style W fill:#B0C4DE
```


## Prediction Flow

1. **`lambda_handler()`**
   AWS Lambda entry point for incoming requests

2. **`predict_fraud()`**
   Coordinates feature preparation and model prediction

3. **`prepare_data_for_model()`**
   Ensures consistent feature encoding with training data

```mermaid
%%{init: {'theme': 'neutral', 'themeVariables': { 'primaryColor': '#F0F8FF', 'edgeLabelBackground':'#FFFFF0'}}}%%
flowchart TD
    A[AWS Lambda Invocation] --> B[Lambda Handler]
    B --> C{Parse Input}
    C -->|JSON String| D[Convert to Dict]
    C -->|Direct Dict| D
    D --> E[Create DataFrame]
    E --> F["Preprocess Data:
    - Drop 'status' column
    - One-hot encode categories
    - Extract hour/day features"]
    F --> G["Feature Alignment:
    - Load feature_names.json
    - Reindex columns
    - Fill missing with 0"]
    G --> H["Load Assets:
    - Trained pipeline (joblib)
    - Feature names"]
    H --> I[Make Prediction]
    I --> J["Return Results:
    - is_fraud (0/1)
    - fraud_probability"]
    J --> K[API Response]

    style A fill:#FF9900,color:black
    style B fill:#4682B4,color:white
    style F fill:#B0C4DE
    style G fill:#B0C4DE
    style H fill:#B0C4DE
    style I fill:#32CD32,color:black
    style J fill:#B0C4DE
    style K fill:#228B22,color:white

    subgraph "Model Serving Components"
    H
    I
    end

    subgraph "Data Processing"
    F
    G
    end
```

## Prediction Local Run

1. **Prerequisites**
   - Python 3.8+
2. **Setup**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run**
   You can edit the `transaction_json` to test different transactions.
   ```bash
   python fraud_prediction_lambda.py
   ```
