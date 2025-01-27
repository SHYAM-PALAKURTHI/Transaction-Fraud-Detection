
# SSENSE Take Home Assignment

## Introduction
This project is a conceptual implementation of a real-time fraud detection pipeline for an e-commerce platform. It simulates the ingestion, processing, and analysis of transaction data to flag potentially fraudulent activities using AWS services, machine learning, and Infrastructure as Code (IaC).

---

## Objectives
1. **Pipeline Design**: Architect a real-time fraud detection system using AWS services.
2. **Platform Engineering**: Automate infrastructure setup with Terraform and CI/CD workflows (Github Actions).
3. **Machine Learning Workflow**: Train and deploy a fraud detection model with feature engineering.
4. **Integration**: Simulate transaction ingestion and real-time predictions.
5. **Documentation**: Explain trade-offs, design decisions, and steps to run the system.

---

## Repository Structure
```
/
├── .github/
│   └── workflows/
│       └── deploy.yml
├── data/
│   ├── original_transactions.csv
│   ├── X_test_original.csv
│   ├── X_train_original.csv
│   ├── X_train_resampled.csv
│   ├── y_test_original.csv
│   ├── y_train_original.csv
│   └── y_train_resampled.csv
├── docs/
│   ├── code-walkthrough.md
│   ├── conceptual-design.md
│   ├── deployment-guide.md
│   ├── diagrams.md
│   ├── fraud_detection_prototype.ipynb
│   └── images/
│       └──  FraudDetectionAWSLambda.png
│       └──  HighLevelArchitectureDiagram.png
│       └──  ModelTraining.png
├── infra/
│   ├── main.tf
│   ├── outputs.tf
│   └── variables.tf
├── models/
│   ├── feature_names.json -> Feature names to use for prediction
│   ├── fraud_detection_pipeline_smote.joblib -> Final model
│   └── fraud_model_evaluation_smote.png -> Model Evaluation 
├── fraud_prediction_lambda.py
├── .gitignore
├── LICENSE
├── Readme.md
├── requirements.txt
└── train_fraud_detection_model.py
```

---

## Key Components
| Component              | Description                                                                 | Documentation Links |
|------------------------|-----------------------------------------------------------------------------|----------------------|
| **Conceptual Design**  | High-level architecture, trade-offs, and AWS service selection.             | [Conceptual Design](docs/conceptual-design.md), [Diagrams](docs/diagrams.md) |
| **Infrastructure**     | Terraform scripts for S3, Lambda, and DynamoDB.                             | [Infrastructure Code](infra/), [Deployment Guide](docs/deployment-guide.md) |
| **ML Model**           | Logistic Regression model trained on resampled data to handle class imbalance. | [Code Walkthrough](docs/code-walkthrough.md), [Jupyter Notebook](docs/fraud_detection_prototype.ipynb), [Training Script](train_fraud_detection_model.py) [Prediction Script](fraud_prediction_lambda.py) |
| **CI/CD**              | GitHub Actions workflow for automated deployment.                           | [Deployment Workflow](.github/workflows/deploy.yml) |
| **Code Walkthrough**   | Explanation of key scripts and Lambda function logic.                       | [Code Walkthrough](docs/code-walkthrough.md) |

---


## Technologies Used
- **AWS**: Kinesis, Lambda, S3, DynamoDB
- **IaC**: Terraform
- **ML**: Python, scikit-learn, SMOTE for resampling
- **CI/CD**: GitHub Actions

---


