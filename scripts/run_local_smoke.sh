#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-sample}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESOURCES="${CLLLM_RESOURCES:-$HOME/Documents/Clllm_resources}"
DATA_ROOT="${CLLLM_DATA_ROOT:-$RESOURCES/shared_data}"
MODEL_PATH="${CLLLM_MODEL_PATH:-$RESOURCES/shared_models/PubMedBERT}"
RUNS_ROOT="${CLLLM_RUNS_ROOT:-$RESOURCES/runs}"
CONDA_ENV="${CLLLM_CONDA_ENV:-deepke-llm}"
DEVICE="${CLLLM_DEVICE:-cpu}"

mkdir -p "$RUNS_ROOT"
cd "$REPO_ROOT"

case "$MODE" in
  sample)
    conda run -n "$CONDA_ENV" python projects/ner_span_pruner/run.py \
      --model-name "$MODEL_PATH" \
      --output-dir "$RUNS_ROOT/ner_sample_smoke" \
      --device "$DEVICE" --smoke-test --local-files-only

    conda run -n "$CONDA_ENV" python projects/el_rag_contrastive/run.py \
      --model-name "$MODEL_PATH" \
      --output-dir "$RUNS_ROOT/el_sample_smoke" \
      --device "$DEVICE" --smoke-test --local-files-only
    ;;
  real)
    conda run -n "$CONDA_ENV" python projects/ner_span_pruner/run.py \
      --dict-train "$DATA_ROOT/bc5cdr-DictMatching/train.txt" \
      --chatgpt-train "$DATA_ROOT/bc5cdr-ChatGPT/train.txt" \
      --gold-dev "$DATA_ROOT/data_pre/data_pred/BC5CDR/merged_bc5cdr_development_data.json" \
      --gold-test "$DATA_ROOT/data_pre/data_pred/BC5CDR/merged_bc5cdr_test_data.json" \
      --model-name "$MODEL_PATH" \
      --output-dir "$RUNS_ROOT/ner_real_smoke" \
      --device "$DEVICE" --smoke-test --local-files-only

    conda run -n "$CONDA_ENV" python projects/el_rag_contrastive/run.py \
      --dataset-dir "$DATA_ROOT/ICD10-ICD11" \
      --model-name "$MODEL_PATH" \
      --output-dir "$RUNS_ROOT/el_real_smoke" \
      --device "$DEVICE" --smoke-test --local-files-only
    ;;
  *)
    echo "Usage: $0 [sample|real]" >&2
    exit 2
    ;;
esac
