#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${PYTHON_EXE:-python}"
EXCEL_PATH="${EXCEL_PATH:-examples/tabular/datasets/mi_core6_ascii.xlsx}"
OUT_DIR="${OUT_DIR:-examples/tabular/outputs_mi_patient_core/mi_core6_ascii}"
PROJECT_NAME="${PROJECT_NAME:-MI-core6-ascii}"

if [[ ! -f "$EXCEL_PATH" ]]; then
  echo "Dataset not found: $EXCEL_PATH"
  echo "Set EXCEL_PATH to your xlsx file and retry."
  exit 1
fi

"$PYTHON_EXE" -m pip install -r "examples/tabular/competition_submit_package/05_deploy_info/requirements-deploy.txt"

"$PYTHON_EXE" "examples/tabular/project_multidisease_rigorous/run_selected_sheets_mlp_transformer.py" \
  --excel "$EXCEL_PATH" \
  --out "$OUT_DIR" \
  --seed 42 \
  --epochs 24 \
  --project_name "$PROJECT_NAME" \
  --label_column "label" \
  --data_sheet "Sheet1" \
  --label_values "C1,C2,C3,C4,C5,C6" \
  --exclude_keywords "name,gender,age,height,weight" \
  --leakage_guard_mode balanced \
  --report_mode on \
  --enable_label_noise_filter 1 \
  --enable_bootstrap_augmentation 1 \
  --augment_target_ratio 1.1 \
  --class_weight_power 1.0 \
  --sampler_weight_power 0.85 \
  --strategy_profile mixed_best
