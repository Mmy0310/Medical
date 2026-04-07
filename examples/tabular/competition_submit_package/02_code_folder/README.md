# 多病种严谨系统（核心版）

本目录已收敛为一条主链路：

- 主训练入口：`run_selected_sheets_mlp_transformer.py`
- 批量编排入口：`run_batch_excels.py`
- 动态索引工具：`discover_outputs.py`
- 报告策略规则：`report_mode_policy.py`

目标是避免重复实现，保证功能完整同时减少维护成本。

## 快速运行

建议使用仓库内虚拟环境：

```bash
./transformer/Scripts/python.exe examples/tabular/project_multidisease_rigorous/run_selected_sheets_mlp_transformer.py \
  --excel "心血管疾病患者生化指标分析.xlsx" \
  --out "examples/tabular/outputs_cardiovascular_biochem_best_v3/心血管疾病患者生化指标分析" \
  --data_sheet "Sheet1" \
  --label_column "出院诊断" \
  --leakage_guard_mode balanced \
  --report_mode on
```

二分类子任务（同一表中限定两个标签）：

```bash
./transformer/Scripts/python.exe examples/tabular/project_multidisease_rigorous/run_selected_sheets_mlp_transformer.py \
  --excel "心血管疾病患者生化指标分析.xlsx" \
  --out "examples/tabular/outputs_cardiovascular_biochem_90fast/心血管疾病患者生化指标分析_二分类90" \
  --data_sheet "Sheet1" \
  --label_column "出院诊断" \
  --label_values "1.急性心力衰竭,1.急性广泛前壁心肌梗死" \
  --report_mode on
```

批量模式（多个 Excel）：

```bash
./transformer/Scripts/python.exe examples/tabular/project_multidisease_rigorous/run_batch_excels.py \
  --input_glob "examples/tabular/datasets/*.xlsx" \
  --out_base "examples/tabular/outputs_batch_quality_full" \
  --leaderboard_metric robust_composite \
  --robust_f1_weight 0.7 \
  --extra_args "--leakage_guard_mode balanced --report_mode on"
```

## 输出文件（单次运行常见）

- `summary.json`：完整配置、清洗记录、模型指标
- `report.md`：汇总报告
- `clinical_dossier_by_disease.md`：病种解释文档
- `feature_importance_*.csv`：特征重要性明细
- `predictions_*.csv`：预测结果
- `imbalance_diagnostics.json`：不均衡诊断
- `model_lightweight_summary.json`：模型轻量化评估

## 输出目录动态索引

```bash
./transformer/Scripts/python.exe examples/tabular/project_multidisease_rigorous/discover_outputs.py \
  --base-dir examples/tabular \
  --prefix outputs_ \
  --format md \
  --write examples/tabular/OUTPUTS_DYNAMIC_INDEX.md
```

## 严谨性说明

- 建议始终开启 `--report_mode on`，保留可发布性与风险分层信息。
- 医疗场景默认建议 `--leakage_guard_mode balanced` 或更严格模式。
- 若有患者标识列，建议启用按患者分组切分，避免同人泄漏。

## 代码精简说明

为消除重叠实现，以下旧入口已移除：

- `run_experiment.py`
- `run_single_disease_mlp_transformer.py`
- `run_multiseed_stability_screen.py`
- `run_upgrade_compare_formal.py`

当前仅维护核心主链路（训练 + 批量 + 报告治理）。
