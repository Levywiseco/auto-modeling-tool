#!/bin/bash
set -euo pipefail

# Run from the repository root:
#   bash scripts/run_pipeline.sh --input data.csv --target bad_flag
python -m auto_modeling_tool.main "$@"
