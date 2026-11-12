#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION, for example us-east-1}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID to your 12-digit AWS account ID}"
: "${SAGEMAKER_EXECUTION_ROLE_ARN:?Set SAGEMAKER_EXECUTION_ROLE_ARN to a SageMaker execution role ARN}"
: "${MODEL_DATA_URL:?Set MODEL_DATA_URL to the s3:// URI of the uploaded model.tar.gz}"
: "${BATCH_INPUT_S3_URI:?Set BATCH_INPUT_S3_URI to the S3 prefix containing JSON Lines input files}"
: "${BATCH_OUTPUT_S3_URI:?Set BATCH_OUTPUT_S3_URI to an S3 prefix for results}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPOSITORY_NAME="${ECR_REPOSITORY_NAME:-auto-pd-api}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "${ROOT_DIR}" rev-parse --short HEAD)}"
INSTANCE_TYPE="${BATCH_INSTANCE_TYPE:-ml.m5.large}"
INSTANCE_COUNT="${BATCH_INSTANCE_COUNT:-1}"
MAX_CONCURRENT="${BATCH_MAX_CONCURRENT_TRANSFORMS:-1}"
MAX_PAYLOAD_MB="${BATCH_MAX_PAYLOAD_MB:-6}"
SUFFIX="$(date -u +%Y%m%d%H%M%S)"
MODEL_NAME="auto-pd-batch-${SUFFIX}"
JOB_NAME="auto-pd-transform-${SUFFIX}"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPOSITORY_NAME}:${IMAGE_TAG}"

aws sagemaker create-model \
    --model-name "${MODEL_NAME}" \
    --execution-role-arn "${SAGEMAKER_EXECUTION_ROLE_ARN}" \
    --primary-container "Image=${IMAGE_URI},ModelDataUrl=${MODEL_DATA_URL},Environment={MODEL_ARTIFACTS_DIR=/opt/ml/model}" \
    --region "${AWS_REGION}" >/dev/null

aws sagemaker create-transform-job \
    --transform-job-name "${JOB_NAME}" \
    --model-name "${MODEL_NAME}" \
    --transform-input "DataSource={S3DataSource={S3DataType=S3Prefix,S3Uri=${BATCH_INPUT_S3_URI}},ContentType=application/json,SplitType=Line}" \
    --transform-output "S3OutputPath=${BATCH_OUTPUT_S3_URI},Accept=application/json,AssembleWith=Line" \
    --transform-resources "InstanceType=${INSTANCE_TYPE},InstanceCount=${INSTANCE_COUNT}" \
    --max-concurrent-transforms "${MAX_CONCURRENT}" \
    --max-payload-in-mb "${MAX_PAYLOAD_MB}" \
    --region "${AWS_REGION}" >/dev/null

echo "Started Batch Transform job ${JOB_NAME}."
echo "Read results under ${BATCH_OUTPUT_S3_URI} after the job completes."
echo "Job status: aws sagemaker describe-transform-job --transform-job-name ${JOB_NAME} --region ${AWS_REGION}"
echo "Batch Transform instances are billed while the job runs; the job does not leave a persistent endpoint running."
