provider "aws" {
  region = var.aws_region
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 Buckets
resource "aws_s3_bucket" "raw_data" {
  bucket = "fraud-raw-${random_id.bucket_suffix.hex}"
  force_destroy = true
}

resource "aws_s3_bucket" "quarantine" {
  bucket = "fraud-quarantine-${random_id.bucket_suffix.hex}"
  force_destroy = true
}

resource "aws_s3_bucket" "model_bucket" {
  bucket = "fraud-model-${random_id.bucket_suffix.hex}"
  force_destroy = true
  versioning {
    enabled = true
  }
}

# DynamoDB Table
resource "aws_dynamodb_table" "user_profiles" {
  name         = "user_profiles"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "user_id"

  attribute {
    name = "user_id"
    type = "S"
  }
}

# Lambda Layer
resource "aws_lambda_layer_version" "dependencies" {
  filename            = "${path.module}/../dependencies.zip"
  layer_name          = "fraud-dependencies"
  compatible_runtimes = ["python3.9"]
  source_code_hash    = filebase64sha256("${path.module}/../dependencies.zip")
}

# Lambda Package
data "archive_file" "lambda_code" {
  type        = "zip"
  source_dir  = "${path.module}/.."
  output_path = "${path.module}/lambda_pkg.zip"
  excludes    = ["data", "models", "*.ipynb", "requirements.txt", "infra/**", ".git/**"]
}

# Lambda Function
resource "aws_lambda_function" "fraud_detection" {
  function_name    = "fraud-detection"
  role             = aws_iam_role.lambda_role.arn
  handler          = "fraud_prediction_lambda.lambda_handler"
  runtime          = "python3.9"
  filename         = data.archive_file.lambda_code.output_path
  source_code_hash = data.archive_file.lambda_code.output_base64sha256
  layers           = [aws_lambda_layer_version.dependencies.arn]
  timeout          = 30
  memory_size      = 512

  environment {
    variables = {
      MODEL_BUCKET     = aws_s3_bucket.model_bucket.id
      DYNAMODB_TABLE   = aws_dynamodb_table.user_profiles.name
      SNS_TOPIC_ARN    = aws_sns_topic.alerts.arn
    }
  }
}

# IAM Role for Lambda
resource "aws_iam_role" "lambda_role" {
  name = "fraud-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_permissions" {
  name = "fraud-lambda-permissions"
  role = aws_iam_role.lambda_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = [
          "${aws_s3_bucket.raw_data.arn}/*",
          "${aws_s3_bucket.quarantine.arn}/*",
          "${aws_s3_bucket.model_bucket.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:UpdateItem"
        ]
        Resource = aws_dynamodb_table.user_profiles.arn
      },
      {
        Effect = "Allow"
        Action = "sns:Publish"
        Resource = aws_sns_topic.alerts.arn
      }
    ]
  })
}

# SNS Topic
resource "aws_sns_topic" "alerts" {
  name = "fraud-alerts"
}

resource "aws_kinesis_stream" "transaction_stream" {
  name             = "fraud-transaction-stream"
  shard_count      = 1
  retention_period = 24
}

output "model_bucket_name" {
  value = aws_s3_bucket.model_bucket.id
}
