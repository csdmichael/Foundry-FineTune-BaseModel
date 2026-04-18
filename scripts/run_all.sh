#!/usr/bin/env bash
set -euo pipefail

python scripts/generate_kla_data.py
python scripts/build_experiment_sets.py
python scripts/upload_to_blob.py
python scripts/run_finetune_experiments.py
python scripts/compare_experiment_results.py
