#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARTIFACTS_DIR="${ROOT_DIR}/artifacts"
OUTPUT_DIR="${ROOT_DIR}/deployment/aws/build"
PACKAGE_PATH="${OUTPUT_DIR}/model.tar.gz"

for artifact in pd_model.joblib model_metadata.json input_schema.json monitoring_profile.json; do
    if [[ ! -f "${ARTIFACTS_DIR}/${artifact}" ]]; then
        echo "Missing ${ARTIFACTS_DIR}/${artifact}; run ./run_pipeline.sh first." >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"
tar -czf "${PACKAGE_PATH}" -C "${ARTIFACTS_DIR}" \
    pd_model.joblib model_metadata.json input_schema.json monitoring_profile.json

echo "Created SageMaker model package: ${PACKAGE_PATH}"
