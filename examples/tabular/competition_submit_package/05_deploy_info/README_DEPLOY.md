# Deploy Quickstart

This folder contains a minimal deployment bundle for the tabular MI classification pipeline.

## Files

- `requirements-deploy.txt`: minimal Python dependencies
- `run_windows.ps1`: one-command run on Windows
- `run_linux.sh`: one-command run on Linux/macOS
- `git_commit_windows.ps1`: one-command whitelist git commit on Windows
- `git_commit_linux.sh`: one-command whitelist git commit on Linux/macOS
- `DEPLOY_LINKS_TEMPLATE.txt`: form-friendly deployment link template

## 1) Git upload scope (recommended)

Upload these folders as a deployment-ready code package:

- `examples/tabular/competition_submit_package/02_code_folder`
- `examples/tabular/competition_submit_package/05_deploy_info`

Optional for reviewers:

- `examples/tabular/competition_submit_package/03_docs_folder`
- `examples/tabular/competition_submit_package/04_demo_folder`

## 2) Run on Windows

```powershell
powershell -ExecutionPolicy Bypass -File examples/tabular/competition_submit_package/05_deploy_info/run_windows.ps1
```

Use custom dataset path if needed:

```powershell
powershell -ExecutionPolicy Bypass -File examples/tabular/competition_submit_package/05_deploy_info/run_windows.ps1 -ExcelPath "path/to/your.xlsx"
```

## 3) Run on Linux/macOS

```bash
chmod +x examples/tabular/competition_submit_package/05_deploy_info/run_linux.sh
./examples/tabular/competition_submit_package/05_deploy_info/run_linux.sh
```

Use custom dataset path if needed:

```bash
EXCEL_PATH="path/to/your.xlsx" ./examples/tabular/competition_submit_package/05_deploy_info/run_linux.sh
```

## 4) Expected output

By default outputs are written to:

- `examples/tabular/outputs_mi_patient_core/mi_core6_ascii`

Key files to check:

- `summary.json`
- `report.md`
- `EXTENDED_METRICS.md`

## 5) One-command Git commit (whitelist)

Windows (required paths only):

```powershell
powershell -ExecutionPolicy Bypass -File examples/tabular/competition_submit_package/05_deploy_info/git_commit_windows.ps1
```

Windows (include docs/demo too):

```powershell
powershell -ExecutionPolicy Bypass -File examples/tabular/competition_submit_package/05_deploy_info/git_commit_windows.ps1 -IncludeOptional
```

Linux/macOS (required paths only):

```bash
chmod +x examples/tabular/competition_submit_package/05_deploy_info/git_commit_linux.sh
./examples/tabular/competition_submit_package/05_deploy_info/git_commit_linux.sh
```

Linux/macOS (include docs/demo too):

```bash
./examples/tabular/competition_submit_package/05_deploy_info/git_commit_linux.sh --include-optional
```
