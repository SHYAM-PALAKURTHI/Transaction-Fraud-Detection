variable "aws_region" {
  description = "AWS region for resource deployment"
  type        = string
  default     = "us-east-1"
}

variable "lambda_timeout" {
  description = "Timeout for Lambda function in seconds"
  type        = number
  default     = 30
}

variable "lambda_memory" {
  description = "Memory allocation for Lambda function (MB)"
  type        = number
  default     = 512
}
data "aws_caller_identity" "current" {}
