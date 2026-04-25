#!/usr/bin/env bash
set -euo pipefail

COMPETITION="sentiment-analysis-on-movie-reviews"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW_DIR="${PROJECT_ROOT}/data/01_raw"

# Load environment variables from .env file
if [ -f "${PROJECT_ROOT}/.env" ]; then
  echo "Loading environment variables from .env..."
  while IFS='=' read -r key value; do
    # Skip comments and empty lines
    if [[ "$key" =~ ^[[:space:]]*# ]] || [[ -z "${key// }" ]]; then
      continue
    fi
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)
    echo "Setting: $key"
    export "$key=$value"
  done < "${PROJECT_ROOT}/.env"
fi

#----------------------------------------------------------
if ! command -v kaggle >/dev/null 2>&1; then
  echo "Error: kaggle CLI not found. Install dependencies first." >&2
  exit 1
fi

if [[ -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ]]; then
  echo "Error: KAGGLE_USERNAME and KAGGLE_KEY must be set as environment variables." >&2
  exit 1
fi

mkdir -p "${RAW_DIR}"

echo "Downloading Kaggle competition files to ${RAW_DIR}..."
kaggle competitions download --competition "${COMPETITION}" --path "${RAW_DIR}" --force

# Extract zips repeatedly until no more zip files remain
while [ "$(find "${RAW_DIR}" -maxdepth 1 -name "*.zip" | wc -l)" -gt 0 ]; do
    echo "Extracting files in ${RAW_DIR}..."
    find "${RAW_DIR}" -maxdepth 1 -type f -name "*.zip" -print0 |
    while IFS= read -r -d '' archive; do
      unzip -o "${archive}" -d "${RAW_DIR}" >/dev/null
      rm -f "${archive}"
    done
done

rm -f ~/.kaggle/kaggle.json

echo "Done. Extracted files are available under ${RAW_DIR}."