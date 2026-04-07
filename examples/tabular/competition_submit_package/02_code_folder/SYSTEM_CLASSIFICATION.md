# 作品提交与代码分类（对齐报名表字段）

本文件把当前工程按“主文件夹/代码/文档/演示/部署”做标准化分类，便于直接填写报名系统。

## 1) 作品主文件夹

- 路径：`examples/tabular/project_multidisease_rigorous/`
- 作用：核心系统目录（训练、批量运行、报告策略、测试）

## 2) 作品文件夹

- 路径：`examples/tabular/`
- 作用：完整项目工作区（数据构建脚本 + 主系统 + 实验输出 + 论文/说明文档）

## 3) 作品代码文件夹

- 主代码：`examples/tabular/project_multidisease_rigorous/`
- 数据构建代码：`examples/tabular/datasets/`

### 3.1 核心训练引擎（Keep）

- `run_selected_sheets_mlp_transformer.py`：通用训练入口（推荐唯一主入口）

### 3.2 编排层（Keep）

- `run_batch_excels.py`：批量 Excel 任务编排
- `run_mixed_profile_multiseed10.py`：多种策略批量复现实验

### 3.3 报告与治理层（Keep）

- `report_mode_policy.py`：报告模式与披露规则
- `discover_outputs.py`：动态发现输出目录并分类
- `make_completed_five_seed_report.py`：多种子汇总报告

### 3.4 数据构建层（Keep）

- `datasets/build_ptbxl_multidisease_zh.py`
- `datasets/build_ptbxl_multidisease_zh_medium.py`
- `datasets/build_mi_binary_highscore_zh.py`
- `datasets/build_mi_like_benchmark_sets.py`
- `datasets/build_mi_rich_multidisease_sets.py`

### 3.5 测试层（Keep）

- `tests/test_batch_and_report_mode.py`

### 3.6 已清理旧代码（Removed）

- `examples/tabular/train_mlp_and_transformer.py`（早期单次原型，已被主系统替代）
- `examples/tabular/project_multidisease_rigorous/run_experiment.py`（与主入口能力重叠）
- `examples/tabular/project_multidisease_rigorous/run_single_disease_mlp_transformer.py`（已可由主入口通过 `--label_values` 覆盖）
- `examples/tabular/project_multidisease_rigorous/run_multiseed_stability_screen.py`（硬编码实验场景）
- `examples/tabular/project_multidisease_rigorous/run_upgrade_compare_formal.py`（依赖历史旧目录）

## 4) 作品文档文件夹

- 建议路径：`examples/tabular/`
- 主文档清单（建议随作品上传）：
  - `PROJECT_ONEPAGE_MAIN.md`
  - `PROJECT_ONEPAGE_APPENDIX.md`
  - `PROJECT_TEST_REPORT_FULL.md`
  - `PROJECT_TEST_REPORT_CH4.md`
  - `REFERENCE_QUICK_LOOKUP.md`
  - `project_multidisease_rigorous/README.md`

## 5) 作品演示文件夹

- 推荐主演示：`examples/tabular/outputs_cardiovascular_biochem_best_v3/`
- 备选演示：`examples/tabular/outputs_batch_quality_full/`
- 内容要求：`report.md`、`clinical_dossier_by_disease.md`、`best_model_winners.json`、关键图表

## 6) 作品部署链接

当前工程为离线科研分析系统，可先填写两种可验证部署形态：

- 部署链接1：代码仓库主页（含 README 与运行命令）
- 部署链接2：演示结果目录或演示视频链接

若后续需要在线部署，建议新增一个轻量 Web 展示层（只读推理与报告查看），并在该文档补充链接。