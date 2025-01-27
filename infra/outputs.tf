output "raw_data_bucket" {
  description = "Raw data S3 bucket name"
  value       = aws_s3_bucket.raw_data.id
}

output "quarantine_bucket" {
  description = "Quarantine S3 bucket name"
  value       = aws_s3_bucket.quarantine.id
}

output "model_bucket" {
  description = "Model storage S3 bucket name"
  value       = aws_s3_bucket.model_bucket.id
}

output "dynamodb_table" {
  description = "DynamoDB table name"
  value       = aws_dynamodb_table.user_profiles.name
}

output "lambda_function_name" {
  description = "Fraud detection Lambda function name"
  value       = aws_lambda_function.fraud_detection.function_name
}

output "sns_topic_arn" {
  description = "SNS alerts topic ARN"
  value       = aws_sns_topic.alerts.arn
}

output "lambda_layer_arn" {
  description = "Lambda layer ARN for dependencies"
  value       = aws_lambda_layer_version.dependencies.arn
}

# IAM
output "lambda_execution_role" {
  value = aws_iam_role.lambda_role.name
}
