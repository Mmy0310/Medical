import argparse
import csv
import json
import statistics
import subprocess
from datetime import datetime
from pathlib import Path


METRIC_KEYS = [
    "mlp_acc",
    "mlp_prec",
    "mlp_rec",
    "mlp_f1",
    "mlp_auc_macro",
    "tr_acc",
    "tr_prec",
    "tr_rec",
    "tr_f1",
    "tr_auc_macro",
    "ens_acc",
    "ens_f1",
    "joint_acc",
    "joint_f1",
]


def as_float(value, default=0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def get_block_metric(block: dict, primary: str, secondary: str) -> float:
    if not isinstance(block, dict):
        return 0.0
    if primary in block:
        return as_float(block.get(primary, 0.0))
    if secondary in block:
        return as_float(block.get(secondary, 0.0))
    return 0.0


def extract_metrics(summary: dict) -> dict:
    if "models" in summary and isinstance(summary["models"], dict):
        mlp = summary["models"].get("mlp", {})
        tr = summary["models"].get("transformer", {})
    else:
        mlp = summary.get("mlp", {})
        tr = summary.get("transformer", {})

    ens = summary.get("ensemble", {})
    joint = summary.get("joint", {})

    metrics = {
        "mlp_acc": get_block_metric(mlp, "accuracy", "test_accuracy"),
        "mlp_prec": get_block_metric(mlp, "macro_precision", "precision"),
        "mlp_rec": get_block_metric(mlp, "macro_recall", "recall"),
        "mlp_f1": get_block_metric(mlp, "macro_f1", "f1"),
        "mlp_auc_macro": get_block_metric(mlp, "roc_auc_macro", "auc_macro"),
        "tr_acc": get_block_metric(tr, "accuracy", "test_accuracy"),
        "tr_prec": get_block_metric(tr, "macro_precision", "precision"),
        "tr_rec": get_block_metric(tr, "macro_recall", "recall"),
        "tr_f1": get_block_metric(tr, "macro_f1", "f1"),
        "tr_auc_macro": get_block_metric(tr, "roc_auc_macro", "auc_macro"),
        "ens_acc": get_block_metric(ens, "accuracy", "test_accuracy"),
        "ens_f1": get_block_metric(ens, "macro_f1", "f1"),
        "joint_acc": get_block_metric(joint, "accuracy", "test_accuracy"),
        "joint_f1": get_block_metric(joint, "macro_f1", "f1"),
    }
    return metrics


def mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    if len(values) == 1:
        return {"mean": float(values[0]), "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def format_pm(mean_value: float, std_value: float) -> str:
    return f"{mean_value:.4f} +- {std_value:.4f}"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--seed_end", type=int, default=51)
    parser.add_argument("--skip_existing", type=int, default=1, choices=[0, 1])
    parser.add_argument("--refresh_outputs_index", type=int, default=0, choices=[0, 1])
    parser.add_argument("--excel", type=str, default="dataset_6diseases.xlsx")
    parser.add_argument(
        "--out_root",
        type=str,
        default="examples/tabular/outputs_project_mixed_profile_multiseed",
    )
    parser.add_argument(
        "--base_seed42_summary",
        type=str,
        default="examples/tabular/outputs_project_multidisease_clinical_v2_multiseed10_protocol/seed_42/summary.json",
    )
    parser.add_argument(
        "--baseline_metrics",
        type=str,
        default="examples/tabular/outputs_project_multidisease_clinical_v2_multiseed10_protocol/metrics_mean_std.json",
    )
    parser.add_argument(
        "--sheets",
        type=str,
        default="",
        help="Comma-separated sheet names; if empty, loaded from base_seed42_summary",
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default="",
        help="Enable single-sheet multiclass mode by providing label column name",
    )
    parser.add_argument(
        "--data_sheet",
        type=str,
        default="",
        help="Sheet name used in single-sheet mode; defaults to first sheet",
    )
    parser.add_argument(
        "--label_values",
        type=str,
        default="",
        help="Optional comma-separated labels to keep in single-sheet mode",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="",
        help="Optional project title for markdown reports",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    python_exe = repo_root / "transformer" / "Scripts" / "python.exe"
    runner = repo_root / "examples" / "tabular" / "project_multidisease_rigorous" / "run_selected_sheets_mlp_transformer.py"

    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found: {python_exe}")
    if not runner.exists():
        raise FileNotFoundError(f"Runner script not found: {runner}")

    excel = (repo_root / args.excel).resolve()
    if not excel.exists():
        raise FileNotFoundError(f"Excel file not found: {excel}")

    label_column = str(args.label_column).strip()
    data_sheet = str(args.data_sheet).strip()
    label_values = str(args.label_values).strip()
    dataset_mode = "multi_sheet"

    if label_column:
        dataset_mode = "single_sheet_label_column"
        selected_sheets = []
        sheets_csv = ""
    else:
        if args.sheets.strip():
            selected_sheets = [s.strip() for s in args.sheets.split(",") if s.strip()]
        else:
            seed42_summary_path = (repo_root / args.base_seed42_summary).resolve()
            if not seed42_summary_path.exists():
                raise FileNotFoundError(f"Base summary not found: {seed42_summary_path}")
            seed42_summary = load_json(seed42_summary_path)
            selected_sheets = list(seed42_summary.get("selected_sheets", []))

        if len(selected_sheets) < 2:
            raise ValueError("Need at least two sheets for multiclass evaluation")
        sheets_csv = ",".join(selected_sheets)

    out_root = (repo_root / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    seeds = list(range(args.seed_start, args.seed_end + 1))

    print(f"[INFO] seeds: {seeds}")
    print(f"[INFO] strategy_profile: mixed_best")
    print(f"[INFO] excel: {excel}")
    print(f"[INFO] dataset_mode: {dataset_mode}")
    if label_column:
        print(f"[INFO] label_column: {label_column}")
        if data_sheet:
            print(f"[INFO] data_sheet: {data_sheet}")
        if label_values:
            print(f"[INFO] label_values: {label_values}")
    else:
        print(f"[INFO] selected_sheets: {selected_sheets}")
    print(f"[INFO] out_root: {out_root}")

    per_seed_records = []
    failed_runs = []

    for seed in seeds:
        seed_out = out_root / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        summary_path = seed_out / "summary.json"
        log_path = seed_out / "train.log"

        run_needed = True
        if bool(args.skip_existing) and summary_path.exists():
            run_needed = False
            print(f"[SKIP] seed={seed}, existing summary found")

        if run_needed:
            cmd = [
                str(python_exe),
                str(runner),
                "--excel",
                str(excel),
                "--out",
                str(seed_out),
                "--seed",
                str(seed),
                "--strategy_profile",
                "mixed_best",
                "--enable_distillation",
                "0",
                "--refresh_outputs_index",
                str(args.refresh_outputs_index),
            ]

            if label_column:
                cmd.extend(["--label_column", label_column])
                if data_sheet:
                    cmd.extend(["--data_sheet", data_sheet])
                if label_values:
                    cmd.extend(["--label_values", label_values])
            else:
                cmd.extend(["--sheets", sheets_csv])

            if str(args.project_name).strip():
                cmd.extend(["--project_name", str(args.project_name).strip()])

            print(f"[RUN] seed={seed}")
            with log_path.open("w", encoding="utf-8") as lf:
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                failed_runs.append(
                    {
                        "seed": seed,
                        "return_code": int(proc.returncode),
                        "log": str(log_path),
                    }
                )
                print(f"[FAIL] seed={seed}, return_code={proc.returncode}")
                continue

        if not summary_path.exists():
            failed_runs.append(
                {
                    "seed": seed,
                    "return_code": -1,
                    "log": str(log_path),
                    "error": "summary.json missing",
                }
            )
            print(f"[FAIL] seed={seed}, summary missing")
            continue

        summary = load_json(summary_path)
        metrics = extract_metrics(summary)
        record = {"seed": seed, **metrics}
        per_seed_records.append(record)
        print(
            "[OK] seed={seed}, mlp_acc={mlp:.4f}, tr_acc={tr:.4f}, joint_acc={joint:.4f}".format(
                seed=seed,
                mlp=record["mlp_acc"],
                tr=record["tr_acc"],
                joint=record["joint_acc"],
            )
        )

    per_seed_records.sort(key=lambda x: x["seed"])

    mean_std_result = {}
    for key in METRIC_KEYS:
        values = [as_float(r.get(key, 0.0)) for r in per_seed_records]
        mean_std_result[key] = mean_std(values)

    seed_count = len(seeds)

    summary_obj = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_profile": "mixed_best",
        "seeds": seeds,
        "seed_count": seed_count,
        "dataset_mode": dataset_mode,
        "label_column": label_column,
        "data_sheet": data_sheet,
        "label_values": label_values,
        "selected_sheets": selected_sheets,
        "excel": str(excel),
        "per_seed": per_seed_records,
        "mean_std": mean_std_result,
        "failed_runs": failed_runs,
    }

    summary_json = out_root / "multiseed_summary.json"
    summary_json.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_seed_json = out_root / f"multiseed{seed_count}_summary.json"
    if summary_seed_json != summary_json:
        summary_seed_json.write_text(json.dumps(summary_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_json = out_root / "metrics_mean_std.json"
    metrics_json.write_text(json.dumps(mean_std_result, ensure_ascii=False, indent=2), encoding="utf-8")

    per_seed_csv = out_root / "metrics_per_seed.csv"
    write_csv(per_seed_csv, per_seed_records, ["seed", *METRIC_KEYS])

    baseline_metrics_path = (repo_root / args.baseline_metrics).resolve()
    baseline_metrics = {}
    delta_rows = []
    if baseline_metrics_path.exists():
        baseline_metrics = load_json(baseline_metrics_path)
        for key in METRIC_KEYS:
            mixed_mean = as_float(mean_std_result.get(key, {}).get("mean", 0.0))
            mixed_std = as_float(mean_std_result.get(key, {}).get("std", 0.0))
            base_mean = as_float(baseline_metrics.get(key, {}).get("mean", 0.0))
            base_std = as_float(baseline_metrics.get(key, {}).get("std", 0.0))
            delta_rows.append(
                {
                    "metric": key,
                    "mixed_mean": mixed_mean,
                    "mixed_std": mixed_std,
                    "old_mean": base_mean,
                    "old_std": base_std,
                    "delta_mean": mixed_mean - base_mean,
                    "delta_std": mixed_std - base_std,
                }
            )

        delta_csv = out_root / "mixed_delta_vs_old.csv"
        write_csv(
            delta_csv,
            delta_rows,
            ["metric", "mixed_mean", "mixed_std", "old_mean", "old_std", "delta_mean", "delta_std"],
        )
    else:
        delta_csv = None

    report_title = str(args.project_name).strip() or "Mixed Best Formal Validation"
    report_lines = []
    report_lines.append(f"# {report_title} ({seed_count}-seed)")
    report_lines.append("")
    report_lines.append("## Protocol")
    report_lines.append(f"- Seeds: {seeds}")
    report_lines.append(f"- Strategy profile: mixed_best")
    report_lines.append(f"- Excel: {excel}")
    report_lines.append(f"- Dataset mode: {dataset_mode}")
    if label_column:
        report_lines.append(f"- Label column: {label_column}")
        if data_sheet:
            report_lines.append(f"- Data sheet: {data_sheet}")
        if label_values:
            report_lines.append(f"- Label values: {label_values}")
    else:
        report_lines.append(f"- Selected sheets: {', '.join(selected_sheets)}")
    report_lines.append(f"- Output: {out_root}")
    report_lines.append("")

    report_lines.append("## Core Metrics (mean +- std)")
    report_lines.append("")
    report_lines.append(f"- MLP Accuracy: {format_pm(mean_std_result['mlp_acc']['mean'], mean_std_result['mlp_acc']['std'])}")
    report_lines.append(f"- MLP Macro-F1: {format_pm(mean_std_result['mlp_f1']['mean'], mean_std_result['mlp_f1']['std'])}")
    report_lines.append(
        f"- MLP ROC-AUC Macro: {format_pm(mean_std_result['mlp_auc_macro']['mean'], mean_std_result['mlp_auc_macro']['std'])}"
    )
    report_lines.append(f"- Transformer Accuracy: {format_pm(mean_std_result['tr_acc']['mean'], mean_std_result['tr_acc']['std'])}")
    report_lines.append(f"- Transformer Macro-F1: {format_pm(mean_std_result['tr_f1']['mean'], mean_std_result['tr_f1']['std'])}")
    report_lines.append(
        f"- Transformer ROC-AUC Macro: {format_pm(mean_std_result['tr_auc_macro']['mean'], mean_std_result['tr_auc_macro']['std'])}"
    )
    report_lines.append(f"- Ensemble Accuracy: {format_pm(mean_std_result['ens_acc']['mean'], mean_std_result['ens_acc']['std'])}")
    report_lines.append(f"- Ensemble Macro-F1: {format_pm(mean_std_result['ens_f1']['mean'], mean_std_result['ens_f1']['std'])}")
    report_lines.append(f"- Joint Accuracy: {format_pm(mean_std_result['joint_acc']['mean'], mean_std_result['joint_acc']['std'])}")
    report_lines.append(f"- Joint Macro-F1: {format_pm(mean_std_result['joint_f1']['mean'], mean_std_result['joint_f1']['std'])}")
    report_lines.append("")

    if baseline_metrics:
        report_lines.append("## Delta vs old baseline")
        report_lines.append("")
        report_lines.append("| Metric | Mixed mean+-std | Old mean+-std | Delta mean |")
        report_lines.append("|---|---:|---:|---:|")
        for key in METRIC_KEYS:
            mm = as_float(mean_std_result.get(key, {}).get("mean", 0.0))
            ms = as_float(mean_std_result.get(key, {}).get("std", 0.0))
            om = as_float(baseline_metrics.get(key, {}).get("mean", 0.0))
            os = as_float(baseline_metrics.get(key, {}).get("std", 0.0))
            report_lines.append(
                f"| {key} | {mm:.4f}+-{ms:.4f} | {om:.4f}+-{os:.4f} | {mm - om:+.4f} |"
            )
        report_lines.append("")

    report_lines.append("## Run status")
    report_lines.append("")
    report_lines.append(f"- Successful runs: {len(per_seed_records)}/{len(seeds)}")
    report_lines.append(f"- Failed runs: {len(failed_runs)}")
    if failed_runs:
        for item in failed_runs:
            report_lines.append(
                f"- Failed seed {item.get('seed')}: return_code={item.get('return_code')}, log={item.get('log')}"
            )

    report_path = out_root / "MIXED_MULTISEED_REPORT.md"
    report_text = "\n".join(report_lines) + "\n"
    report_path.write_text(report_text, encoding="utf-8")
    report_seed_path = out_root / f"MIXED_MULTISEED{seed_count}_REPORT.md"
    if report_seed_path != report_path:
        report_seed_path.write_text(report_text, encoding="utf-8")

    print("\n[ARTIFACTS]")
    print(f"- {summary_json}")
    if summary_seed_json != summary_json:
        print(f"- {summary_seed_json}")
    print(f"- {metrics_json}")
    print(f"- {per_seed_csv}")
    if delta_csv is not None:
        print(f"- {delta_csv}")
    print(f"- {report_path}")
    if report_seed_path != report_path:
        print(f"- {report_seed_path}")

    if failed_runs:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
