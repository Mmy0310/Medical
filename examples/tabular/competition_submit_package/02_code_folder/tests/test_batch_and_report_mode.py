import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import run_batch_excels as batch
import report_mode_policy as policy


def test_parse_extra_args_respects_shell_quotes() -> None:
    parsed = batch.parse_extra_args('--alpha 0.1 --name "foo bar" --flag')
    assert parsed == ["--alpha", "0.1", "--name", "foo bar", "--flag"]


def test_load_run_metrics_extracts_core_fields(tmp_path: Path) -> None:
    summary = {
        "run_config": {
            "leakage_guard_mode": "balanced",
            "report_mode": "on",
            "is_smoke_run": False,
        },
        "report_annotation": {
            "publishability": "publishable_with_disclosure",
            "risk_level": "medium",
            "tier": "balanced_information_retention",
        },
        "rows_total": 1500,
        "dataset_mode": "single_sheet_label_column",
        "n_features_used": {"mlp": 45, "transformer": 45},
        "mlp": {"accuracy": 0.91, "macro_f1": 0.9},
        "transformer": {"accuracy": 0.89, "macro_f1": 0.88},
        "ensemble": {"accuracy": 0.93, "macro_f1": 0.92},
        "joint": {"accuracy": 0.925, "macro_f1": 0.915},
        "imbalance_diagnostics": {
            "ensemble": {
                "worst_class_recall": 0.8,
                "macro_pr_auc": 0.95,
            }
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")

    metrics = batch.load_run_metrics(summary_path)

    assert metrics["summary_found"] is True
    assert metrics["publishability"] == "publishable_with_disclosure"
    assert metrics["ensemble_macro_f1"] == 0.92
    assert metrics["ensemble_worst_class_recall"] == 0.8
    assert metrics["ensemble_macro_pr_auc"] == 0.95


def test_build_leaderboard_supports_robust_composite_ranking() -> None:
    rows = [
        {
            "excel": "A.xlsx",
            "ok": True,
            "dataset_mode": "single_sheet_label_column",
            "rows_total": 100,
            "ensemble_macro_f1": 0.90,
            "ensemble_worst_class_recall": 0.50,
        },
        {
            "excel": "B.xlsx",
            "ok": True,
            "dataset_mode": "single_sheet_label_column",
            "rows_total": 100,
            "ensemble_macro_f1": 0.85,
            "ensemble_worst_class_recall": 0.80,
        },
    ]

    leaderboard = batch.build_leaderboard(rows, metric="robust_composite", robust_f1_weight=0.7)

    assert not leaderboard.empty
    assert leaderboard.iloc[0]["excel_name"] == "B.xlsx"
    assert leaderboard.iloc[0]["robust_composite"] > leaderboard.iloc[1]["robust_composite"]


def test_leaderboard_markdown_deduplicates_dynamic_metric_column() -> None:
    rows = [
        {
            "excel": "C.xlsx",
            "ok": True,
            "dataset_mode": "single_sheet_label_column",
            "rows_total": 120,
            "leakage_guard_mode": "balanced",
            "publishability": "publishable_with_disclosure",
            "ensemble_accuracy": 0.9,
            "ensemble_macro_f1": 0.89,
            "ensemble_worst_class_recall": 0.83,
            "mlp_accuracy": 0.88,
            "transformer_accuracy": 0.87,
        }
    ]

    leaderboard = batch.build_leaderboard(rows, metric="ensemble_macro_f1", robust_f1_weight=0.7)
    md = batch.leaderboard_to_markdown(leaderboard, metric="ensemble_macro_f1", robust_f1_weight=0.7)

    header_line = next(line for line in md.splitlines() if line.startswith("| rank"))
    assert header_line.count("ensemble_macro_f1") == 1


def test_leaderboard_markdown_includes_robust_formula() -> None:
    rows = [
        {
            "excel": "D.xlsx",
            "ok": True,
            "dataset_mode": "single_sheet_label_column",
            "rows_total": 80,
            "ensemble_macro_f1": 0.86,
            "ensemble_worst_class_recall": 0.84,
        }
    ]

    leaderboard = batch.build_leaderboard(rows, metric="robust_composite", robust_f1_weight=0.7)
    md = batch.leaderboard_to_markdown(leaderboard, metric="robust_composite", robust_f1_weight=0.7)

    assert "robust_composite = 0.70" in md


def test_report_mode_annotation_profiles() -> None:
    strict = policy.build_report_mode_annotation(
        report_mode="on",
        leakage_guard_mode="strict",
        smoke_run=False,
    )
    assert strict is not None
    assert strict["publishability"] == "publishable_primary"

    balanced = policy.build_report_mode_annotation(
        report_mode="on",
        leakage_guard_mode="balanced",
        smoke_run=False,
    )
    assert balanced is not None
    assert balanced["publishability"] == "publishable_with_disclosure"

    smoke = policy.build_report_mode_annotation(
        report_mode="on",
        leakage_guard_mode="strict",
        smoke_run=True,
    )
    assert smoke is not None
    assert smoke["tier"] == "smoke_speed_only"

    off_mode = policy.build_report_mode_annotation(
        report_mode="off",
        leakage_guard_mode="balanced",
        smoke_run=False,
    )
    assert off_mode is None
