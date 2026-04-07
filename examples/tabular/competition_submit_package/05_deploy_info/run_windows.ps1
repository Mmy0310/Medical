param(
    [string]$PythonExe = "transformer/Scripts/python.exe",
    [string]$ExcelPath = "examples/tabular/datasets/mi_core6_ascii.xlsx",
    [string]$OutDir = "examples/tabular/outputs_mi_patient_core/mi_core6_ascii",
    [string]$ProjectName = "MI-core6-ascii"
)

if (-not (Test-Path $PythonExe)) {
    Write-Host "Python not found at $PythonExe, fallback to 'python'."
    $PythonExe = "python"
}

if (-not (Test-Path $ExcelPath)) {
    Write-Error "Dataset not found: $ExcelPath"
    Write-Host "Set -ExcelPath to your own xlsx file and retry."
    exit 1
}

& $PythonExe -m pip install -r "examples/tabular/competition_submit_package/05_deploy_info/requirements-deploy.txt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $PythonExe "examples/tabular/project_multidisease_rigorous/run_selected_sheets_mlp_transformer.py" `
  --excel "$ExcelPath" `
  --out "$OutDir" `
  --seed 42 `
  --epochs 24 `
  --project_name "$ProjectName" `
  --label_column "label" `
  --data_sheet "Sheet1" `
  --label_values "C1,C2,C3,C4,C5,C6" `
  --exclude_keywords "name,gender,age,height,weight" `
  --leakage_guard_mode balanced `
  --report_mode on `
  --enable_label_noise_filter 1 `
  --enable_bootstrap_augmentation 1 `
  --augment_target_ratio 1.1 `
  --class_weight_power 1.0 `
  --sampler_weight_power 0.85 `
  --strategy_profile mixed_best

exit $LASTEXITCODE
