#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION, for example us-east-1}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID to your 12-digit AWS account ID}"
: "${SAGEMAKER_EXECUTION_ROLE_ARN:?Set SAGEMAKER_EXECUTION_ROLE_ARN to a SageMaker execution role ARN}"
: "${MODEL_DATA_URL:?Set MODEL_DATA_URL to the s3:// URI of the uploaded model.tar.gz}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPOSITORY_NAME="${ECR_REPOSITORY_NAME:-auto-pd-api}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "${ROOT_DIR}" rev-parse --short HEAD)}"
ENDPOINT_NAME="${SAGEMAKER_ENDPOINT_NAME:-auto-pd-endpoint}"
INSTANCE_TYPE="${SAGEMAKER_INSTANCE_TYPE:-ml.m5.large}"
INSTANCE_COUNT="${SAGEMAKER_INSTANCE_COUNT:-1}"
SUFFIX="$(date -u +%Y%m%d%H%M%S)"
MODEL_NAME="auto-pd-${SUFFIX}"
ENDPOINT_CONFIG_NAME="auto-pd-config-${SUFFIX}"
IMAGE_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPOSITORY_NAME}:${IMAGE_TAG}"

aws sagemaker create-model \
    --model-name "${MODEL_NAME}" \
    --execution-role-arn "${SAGEMAKER_EXECUTION_ROLE_ARN}" \
    --primary-container "Image=${IMAGE_URI},ModelDataUrl=${MODEL_DATA_URL},Environment={MODEL_ARTIFACTS_DIR=/opt/ml/model}" \
    --region "${AWS_REGION}" >/dev/null

aws sagemaker create-endpoint-config \
    --endpoint-config-name "${ENDPOINT_CONFIG_NAME}" \
    --production-variants "VariantName=AllTraffic,ModelName=${MODEL_NAME},InitialInstanceCount=${INSTANCE_COUNT},InstanceType=${INSTANCE_TYPE},InitialVariantWeight=1" \
    --region "${AWS_REGION}" >/dev/null

if aws sagemaker describe-endpoint --endpoint-name "${ENDPOINT_NAME}" \
    --region "${AWS_REGION}" >/dev/null 2>&1; then
    aws sagemaker update-endpoint \
        --endpoint-name "${ENDPOINT_NAME}" \
        --endpoint-config-name "${ENDPOINT_CONFIG_NAME}" \
        --region "${AWS_REGION}" >/dev/null
else
    aws sagemaker create-endpoint \
        --endpoint-name "${ENDPOINT_NAME}" \
        --endpoint-config-name "${ENDPOINT_CONFIG_NAME}" \
        --region "${AWS_REGION}" >/dev/null
fi

echo "Started deployment of ${ENDPOINT_NAME}; endpoint creation may take several minutes."
echo "Check status: aws sagemaker describe-endpoint --endpoint-name ${ENDPOINT_NAME} --region ${AWS_REGION}"
echo "Remember: a running SageMaker endpoint incurs ongoing charges; delete it when finished."
