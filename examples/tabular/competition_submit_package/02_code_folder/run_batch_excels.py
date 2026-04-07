import argparse
import glob
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd


META_SHEET_KEYWORDS = ("说明", "字段", "readme", "README", "note", "备注")
LABEL_CANDIDATES = ("疾病标签", "label", "Label", "类别", "病种", "诊断")
BASE_LEADERBOARD_METRICS = (
    "ensemble_macro_f1",
    "ensemble_accuracy",
    "joint_macro_f1",
    "joint_accuracy",
    "mlp_macro_f1",
    "mlp_accuracy",
    "transformer_macro_f1",
    "transformer_accuracy",
)
COMPUTED_LEADERBOARD_METRICS = (
    "robust_composite",
)
LEADERBOARD_METRICS = BASE_LEADERBOARD_METRICS + COMPUTED_LEADERBOARD_METRICS


def is_meta_sheet(name: str) -> bool:
    n = str(name).strip()
    if not n:
        return True
    return any(k in n for k in META_SHEET_KEYWORDS)


def detect_mode(excel_path: Path, forced_label_column: str, forced_data_sheet: str):
    xls = pd.ExcelFile(excel_path)
    sheet_names = [str(s).strip() for s in xls.sheet_names if str(s).strip()]

    if forced_label_column:
        data_sheet = forced_data_sheet.strip() if forced_data_sheet else sheet_names[0]
        return {
            "mode": "single_sheet_label_column",
            "data_sheet": data_sheet,
            "label_column": forced_label_column,
            "sheets": [],
        }

    for s in sheet_names:
        if is_meta_sheet(s):
            continue
        try:
            head = pd.read_excel(excel_path, sheet_name=s, nrows=1)
        except Exception:
            continue
        for cand in LABEL_CANDIDATES:
            if cand in head.columns:
                return {
                    "mode": "single_sheet_label_column",
                    "data_sheet": s,
                    "label_column": cand,
                    "sheets": [],
                }

    usable_sheets = [s for s in sheet_names if not is_meta_sheet(s)]
    if len(usable_sheets) < 2:
        raise ValueError(
            f"Cannot infer mode for {excel_path}. Provide --label_column or ensure >=2 non-meta sheets."
        )

    return {
        "mode": "multi_sheet",
        "data_sheet": "",
        "label_column": "",
        "sheets": usable_sheets,
    }


def build_command(args, excel_path: Path, out_dir: Path):
    mode_info = detect_mode(
        excel_path=excel_path,
        forced_label_column=args.label_column,
        forced_data_sheet=args.data_sheet,
    )

    cmd = [
        sys.executable,
        str(Path(args.runner).resolve()),
        "--excel",
        str(excel_path.resolve()),
        "--out",
        str(out_dir.resolve()),
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--project_name",
        f"{args.project_name_prefix}{excel_path.stem}",
    ]

    if mode_info["mode"] == "single_sheet_label_column":
        cmd += [
            "--label_column",
            mode_info["label_column"],
            "--data_sheet",
            mode_info["data_sheet"],
        ]
        if args.label_values:
            cmd += ["--label_values", args.label_values]
    else:
        if args.sheets:
            sheet_arg = args.sheets
        else:
            sheet_arg = ",".join(mode_info["sheets"])
        cmd += ["--sheets", sheet_arg]

    if args.exclude_keywords:
        cmd += ["--exclude_keywords", args.exclude_keywords]

    if args.extra_args:
        cmd += args.extra_args

    return cmd, mode_info


def safe_float(x):
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None


def pick_metric(d: dict, keys: list[str]):
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d:
            v = safe_float(d.get(k))
            if v is not None:
                return v
    return None


def load_run_metrics(summary_path: Path) -> dict:
    out = {
        "summary_found": False,
        "summary_error": "",
    }
    if not summary_path.exists():
        out["summary_error"] = "summary.json not found"
        return out

    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as ex:
        out["summary_error"] = f"failed to parse summary.json: {type(ex).__name__}: {ex}"
        return out

    out["summary_found"] = True
    out["rows_total"] = data.get("rows_total")
    out["dataset_mode"] = data.get("dataset_mode")

    run_config = data.get("run_config", {}) if isinstance(data.get("run_config", {}), dict) else {}
    out["leakage_guard_mode"] = run_config.get("leakage_guard_mode", "")
    out["report_mode"] = run_config.get("report_mode", "")
    out["is_smoke_run"] = bool(run_config.get("is_smoke_run", False))

    report_annotation = data.get("report_annotation", {}) if isinstance(data.get("report_annotation", {}), dict) else {}
    out["publishability"] = report_annotation.get("publishability", "")
    out["risk_level"] = report_annotation.get("risk_level", "")
    out["report_tier"] = report_annotation.get("tier", "")

    n_features_used = data.get("n_features_used", {}) if isinstance(data.get("n_features_used", {}), dict) else {}
    out["n_features_mlp"] = n_features_used.get("mlp")
    out["n_features_transformer"] = n_features_used.get("transformer")

    imbalance = data.get("imbalance_diagnostics", {}) if isinstance(data.get("imbalance_diagnostics", {}), dict) else {}
    ensemble_imbalance = imbalance.get("ensemble", {}) if isinstance(imbalance.get("ensemble", {}), dict) else {}
    out["ensemble_worst_class_recall"] = pick_metric(
        ensemble_imbalance,
        ["worst_class_recall"],
    )
    out["ensemble_macro_pr_auc"] = pick_metric(
        ensemble_imbalance,
        ["macro_pr_auc"],
    )

    model_scopes = {
        "mlp": [data.get("mlp", {}), data.get("models", {}).get("mlp", {}) if isinstance(data.get("models", {}), dict) else {}],
        "transformer": [
            data.get("transformer", {}),
            data.get("models", {}).get("transformer", {}) if isinstance(data.get("models", {}), dict) else {},
        ],
        "ensemble": [
            data.get("ensemble", {}),
            data.get("models", {}).get("ensemble", {}) if isinstance(data.get("models", {}), dict) else {},
        ],
        "joint": [data.get("joint", {}), data.get("models", {}).get("joint", {}) if isinstance(data.get("models", {}), dict) else {}],
    }

    for model_name, scopes in model_scopes.items():
        acc = None
        f1 = None
        for scope in scopes:
            if not isinstance(scope, dict):
                continue
            acc = acc if acc is not None else pick_metric(scope, ["accuracy", "test_accuracy"])
            f1 = f1 if f1 is not None else pick_metric(scope, ["macro_f1", "test_macro_f1"])
        out[f"{model_name}_accuracy"] = acc
        out[f"{model_name}_macro_f1"] = f1

    return out


def build_leaderboard(rows: list[dict], metric: str, robust_f1_weight: float) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if "excel" in df.columns:
        df["excel_name"] = df["excel"].map(lambda x: Path(str(x)).name)

    for col in LEADERBOARD_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ensemble_macro_f1" in df.columns:
        ensemble_f1 = pd.to_numeric(df["ensemble_macro_f1"], errors="coerce")
    else:
        ensemble_f1 = pd.Series([math.nan] * len(df), index=df.index)

    if "ensemble_worst_class_recall" in df.columns:
        ensemble_worst_recall = pd.to_numeric(df["ensemble_worst_class_recall"], errors="coerce")
    else:
        ensemble_worst_recall = pd.Series([math.nan] * len(df), index=df.index)

    robust = robust_f1_weight * ensemble_f1 + (1.0 - robust_f1_weight) * ensemble_worst_recall
    robust = robust.where(ensemble_f1.notna() & ensemble_worst_recall.notna(), ensemble_f1)
    df["robust_composite"] = robust

    ok_mask = df["ok"].astype(bool) if "ok" in df.columns else pd.Series([True] * len(df), index=df.index)
    rank_df = df.loc[ok_mask].copy()
    if metric in rank_df.columns:
        rank_df = rank_df[rank_df[metric].notna()].copy()

    if rank_df.empty:
        return rank_df

    rank_df = rank_df.sort_values(by=[metric, "excel_name"], ascending=[False, True]).reset_index(drop=True)
    rank_df.insert(0, "rank", range(1, len(rank_df) + 1))
    return rank_df


def leaderboard_to_markdown(df: pd.DataFrame, metric: str, robust_f1_weight: float) -> str:
    lines = ["# 批量多表自动排行榜\n\n"]
    lines.append(f"- 排序指标: {metric}\n")
    if metric == "robust_composite":
        recall_weight = 1.0 - robust_f1_weight
        lines.append(
            f"- 复合得分公式: robust_composite = {robust_f1_weight:.2f} * ensemble_macro_f1 + {recall_weight:.2f} * ensemble_worst_class_recall\n"
        )
    lines.append(f"- 参评任务数: {len(df)}\n\n")

    if df.empty:
        lines.append("无可用结果（可能均失败或未产出summary）。\n")
        return "".join(lines)

    cols = [
        "rank",
        "excel_name",
        "dataset_mode",
        "rows_total",
        "leakage_guard_mode",
        "publishability",
        metric,
        "robust_composite",
        "ensemble_worst_class_recall",
        "ensemble_accuracy",
        "ensemble_macro_f1",
        "mlp_accuracy",
        "transformer_accuracy",
    ]
    seen = set()
    use_cols: list[str] = []
    for c in cols:
        if c in df.columns and c not in seen:
            seen.add(c)
            use_cols.append(c)
    lines.append("| " + " | ".join(use_cols) + " |\n")
    lines.append("|" + "|".join(["---"] * len(use_cols)) + "|\n")
    for row in df[use_cols].itertuples(index=False):
        vals = ["" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(vals) + " |\n")
    lines.append("\n")
    return "".join(lines)


def parse_extra_args(extra: str) -> list[str]:
    if not extra.strip():
        return []
    return shlex.split(extra.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", type=str, default="examples/tabular/datasets/*.xlsx")
    parser.add_argument("--out_base", type=str, default="examples/tabular/outputs_batch_multidisease")
    parser.add_argument(
        "--runner",
        type=str,
        default="examples/tabular/project_multidisease_rigorous/run_selected_sheets_mlp_transformer.py",
    )
    parser.add_argument("--label_column", type=str, default="")
    parser.add_argument("--data_sheet", type=str, default="")
    parser.add_argument("--label_values", type=str, default="")
    parser.add_argument("--sheets", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--project_name_prefix", type=str, default="批量分析-")
    parser.add_argument(
        "--exclude_keywords",
        type=str,
        default="姓名,名字,性别,年龄,身高,体重,Name,Gender,Age,Height,Weight",
    )
    parser.add_argument(
        "--extra_args",
        type=str,
        default="",
        help="Extra args appended to runner. Example: --extra_args \"--topk 15 --perm_repeats 1\"",
    )
    parser.add_argument(
        "--leaderboard_metric",
        type=str,
        default="ensemble_macro_f1",
        choices=list(LEADERBOARD_METRICS),
        help="Metric used for automatic leaderboard sorting",
    )
    parser.add_argument(
        "--robust_f1_weight",
        type=float,
        default=0.7,
        help="F1 weight for robust_composite (recall weight = 1 - robust_f1_weight)",
    )
    parser.add_argument("--enable_leaderboard", type=int, default=1, choices=[0, 1])
    parser.add_argument("--dry_run", type=int, default=0, choices=[0, 1])
    args = parser.parse_args()

    args.extra_args = parse_extra_args(args.extra_args)

    if not (0.0 <= float(args.robust_f1_weight) <= 1.0):
        raise ValueError("--robust_f1_weight must be in [0, 1]")

    excel_files = sorted([Path(p) for p in glob.glob(args.input_glob)])
    if not excel_files:
        raise FileNotFoundError(f"No excel files matched: {args.input_glob}")

    out_base = Path(args.out_base)
    out_base.mkdir(parents=True, exist_ok=True)

    summary = []
    for idx, excel_path in enumerate(excel_files, start=1):
        out_dir = out_base / excel_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            cmd, mode_info = build_command(args=args, excel_path=excel_path, out_dir=out_dir)
            print(f"[{idx}/{len(excel_files)}] {excel_path.name} -> mode={mode_info['mode']}")
            if args.dry_run:
                print("DRY RUN:", " ".join(cmd))
                rc = 0
                run_metrics = {"summary_found": False, "summary_error": "dry_run"}
            else:
                rc = subprocess.run(cmd, check=False).returncode
                run_metrics = load_run_metrics(out_dir / "summary.json")

            summary.append(
                {
                    "excel": str(excel_path),
                    "output_dir": str(out_dir),
                    "mode": mode_info["mode"],
                    "return_code": int(rc),
                    "ok": bool(rc == 0),
                    "command": cmd,
                    **run_metrics,
                }
            )
        except Exception as e:
            summary.append(
                {
                    "excel": str(excel_path),
                    "output_dir": str(out_dir),
                    "mode": "unknown",
                    "return_code": -1,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                }
            )

    summary_path = out_base / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if bool(args.enable_leaderboard):
        leaderboard_df = build_leaderboard(
            summary,
            metric=args.leaderboard_metric,
            robust_f1_weight=float(args.robust_f1_weight),
        )
        leaderboard_csv = out_base / "batch_leaderboard.csv"
        leaderboard_json = out_base / "batch_leaderboard.json"
        leaderboard_md = out_base / "BATCH_LEADERBOARD.md"

        leaderboard_df.to_csv(leaderboard_csv, index=False, encoding="utf-8")
        leaderboard_json.write_text(
            json.dumps(leaderboard_df.to_dict(orient="records"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        leaderboard_md.write_text(
            leaderboard_to_markdown(
                leaderboard_df,
                metric=args.leaderboard_metric,
                robust_f1_weight=float(args.robust_f1_weight),
            ),
            encoding="utf-8",
        )

    ok_count = sum(1 for x in summary if x.get("ok"))
    print(f"done: {ok_count}/{len(summary)} succeeded")
    print(f"summary: {summary_path}")
    if bool(args.enable_leaderboard):
        print(f"leaderboard(csv): {out_base / 'batch_leaderboard.csv'}")
        print(f"leaderboard(md): {out_base / 'BATCH_LEADERBOARD.md'}")


if __name__ == "__main__":
    main()
