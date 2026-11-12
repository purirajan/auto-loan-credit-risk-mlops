#!/usr/bin/env bash
set -euo pipefail

: "${AWS_REGION:?Set AWS_REGION, for example us-east-1}"
: "${AWS_ACCOUNT_ID:?Set AWS_ACCOUNT_ID to your 12-digit AWS account ID}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPOSITORY_NAME="${ECR_REPOSITORY_NAME:-auto-pd-api}"
IMAGE_TAG="${IMAGE_TAG:-$(git -C "${ROOT_DIR}" rev-parse --short HEAD)}"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPOSITORY_NAME}:${IMAGE_TAG}"

if ! aws ecr describe-repositories \
    --repository-names "${REPOSITORY_NAME}" \
    --region "${AWS_REGION}" >/dev/null 2>&1; then
    aws ecr create-repository \
        --repository-name "${REPOSITORY_NAME}" \
        --image-scanning-configuration scanOnPush=true \
        --image-tag-mutability IMMUTABLE \
        --region "${AWS_REGION}" >/dev/null
fi

aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${REGISTRY}"
docker buildx build --platform linux/amd64 --load -t "${IMAGE_URI}" "${ROOT_DIR}"
docker push "${IMAGE_URI}"
echo "Pushed ${IMAGE_URI}"
