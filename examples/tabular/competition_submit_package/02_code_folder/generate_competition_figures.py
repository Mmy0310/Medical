from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("d:/transformers-main")
OUT_DIR = ROOT / "examples/tabular/competition_effect_images"
BATCH_CSV = ROOT / "examples/tabular/outputs_batch_mi_binary_highscore/batch_leaderboard.csv"
SUMMARY_JSON = ROOT / "examples/tabular/outputs_batch_mi_binary_highscore/mi_binary_highscore_zh_1000/summary.json"


def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_font() -> None:
    # Prefer common Chinese fonts on Windows; fall back gracefully.
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_summary() -> dict:
    with SUMMARY_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def fig_system_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(14, 4.8))
    ax.axis("off")

    steps = [
        "Excel数据接入",
        "清洗与特征工程",
        "MLP/Transformer训练",
        "融合推理(Ensemble/Joint)",
        "解释与报告输出",
    ]

    x_positions = [0.05, 0.25, 0.46, 0.68, 0.87]
    y = 0.52

    for i, (x, text) in enumerate(zip(x_positions, steps)):
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.5", "fc": "#f2f8ff", "ec": "#2b6cb0", "lw": 1.5},
            transform=ax.transAxes,
        )
        if i < len(x_positions) - 1:
            ax.annotate(
                "",
                xy=(x_positions[i + 1] - 0.06, y),
                xytext=(x + 0.06, y),
                arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#2b6cb0"},
                xycoords=ax.transAxes,
            )

    ax.text(
        0.5,
        0.9,
        "图1 系统总体流程图（心谱智析）",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        transform=ax.transAxes,
    )

    ax.text(
        0.5,
        0.18,
        "支持单文件与批量模式，输出summary/report/leaderboard及可解释性文件",
        ha="center",
        va="center",
        fontsize=11,
        color="#444",
        transform=ax.transAxes,
    )

    save(fig, "fig01_system_pipeline.png")


def fig_batch_leaderboard() -> None:
    df = pd.read_csv(BATCH_CSV)
    df = df.sort_values("robust_composite", ascending=False).copy()
    df["name"] = df["excel_name"].str.replace(".xlsx", "", regex=False)

    fig, ax1 = plt.subplots(figsize=(10.5, 5.8))
    x = range(len(df))

    bars = ax1.bar(x, df["robust_composite"], color="#3182ce", alpha=0.9, label="Robust Composite")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["name"], rotation=12)
    ax1.set_ylabel("Robust Composite")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(x, df["ensemble_accuracy"], color="#e53e3e", marker="o", lw=2.0, label="Ensemble Accuracy")
    ax2.set_ylabel("Ensemble Accuracy")
    ax2.set_ylim(0, 1.05)

    for rect in bars:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", fontsize=9)

    fig.suptitle("图2 心肌梗高分批次排行榜", fontsize=14, weight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    save(fig, "fig02_batch_leaderboard.png")


def fig_case_config_compare() -> None:
    summary = load_summary()
    data = pd.DataFrame(
        {
            "mode": ["MLP", "Transformer", "Ensemble", "Joint"],
            "accuracy": [
                float(summary["mlp"]["accuracy"]),
                float(summary["transformer"]["accuracy"]),
                float(summary["ensemble"]["accuracy"]),
                float(summary["joint"]["accuracy"]),
            ],
            "macro_f1": [
                float(summary["mlp"]["macro_f1"]),
                float(summary["transformer"]["macro_f1"]),
                float(summary["ensemble"]["macro_f1"]),
                float(summary["joint"]["macro_f1"]),
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    width = 0.35
    x = range(len(data))

    bars1 = ax.bar([i - width / 2 for i in x], data["accuracy"], width=width, label="Accuracy", color="#2b6cb0")
    bars2 = ax.bar([i + width / 2 for i in x], data["macro_f1"], width=width, label="Macro-F1", color="#38a169")

    for b in list(bars1) + list(bars2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 0.004, f"{h:.3f}", ha="center", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(data["mode"])
    ax.set_ylim(0.9, 1.02)
    ax.set_ylabel("Score")
    ax.legend(loc="upper right")
    ax.set_title("图3 心肌梗高分批次模型指标对比", fontsize=14, weight="bold")

    save(fig, "fig03_case_config_compare.png")


def fig_inference_mode_compare() -> None:
    summary = load_summary()
    batch = pd.read_csv(BATCH_CSV).sort_values("robust_composite", ascending=False).iloc[0]
    robust = float(batch["robust_composite"])
    worst_recall = float(summary["imbalance_diagnostics"]["ensemble"]["worst_class_recall"])
    macro_pr_auc = float(summary["imbalance_diagnostics"]["ensemble"]["macro_pr_auc"])

    data = pd.DataFrame(
        {
            "metric": ["Robust Composite", "Worst-class Recall", "Macro PR-AUC"],
            "score": [robust, worst_recall, macro_pr_auc],
        }
    )

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    bars = ax.bar(data["metric"], data["score"], color=["#dd6b20", "#2b6cb0", "#38a169"], width=0.55)
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2.0, h + 0.004, f"{h:.3f}", ha="center", fontsize=10)

    ax.set_ylim(0.9, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("图4 心肌梗高分批次稳健指标", fontsize=14, weight="bold")

    save(fig, "fig04_inference_mode_compare.png")


def fig_class_distribution() -> None:
    summary = load_summary()

    dist = summary.get("class_distribution_by_label", {})
    items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)

    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(12, 6.8))
    ax.barh(labels, values, color="#805ad5", alpha=0.9)
    ax.invert_yaxis()
    ax.set_xlabel("样本数")
    ax.set_title("图5 心肌梗高分数据集类别分布（二分类）", fontsize=14, weight="bold")

    for i, v in enumerate(values):
        ax.text(v + 6, i, str(v), va="center", fontsize=9)

    save(fig, "fig05_class_distribution.png")


def main() -> None:
    ensure_output_dir()
    setup_font()
    fig_system_pipeline()
    fig_batch_leaderboard()
    fig_case_config_compare()
    fig_inference_mode_compare()
    fig_class_distribution()
    print(f"Generated figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
