# Medical

使用 Transformer 和全连接神经网络对医学案例进行学习推理。

## 项目简介

本项目基于 PyTorch 实现了一个医学案例分类模型，结合 **Transformer 编码器**与**全连接神经网络**，对结构化医学特征数据进行学习与推理。

## 项目结构

```
Medical/
├── configs/
│   └── config.yaml        # 模型与训练超参数配置
├── data/
│   ├── __init__.py
│   └── dataset.py         # 数据集加载与预处理
├── models/
│   ├── __init__.py
│   └── model.py           # MedicalTransformerFC 模型定义
├── scripts/
│   ├── train.py           # 训练脚本
│   └── inference.py       # 推理脚本
├── utils/
│   ├── __init__.py
│   └── utils.py           # 通用工具函数
├── requirements.txt
└── README.md
```

## 环境安装

```bash
pip install -r requirements.txt
```

## 数据格式

数据以 CSV 格式存储，每行为一个样本，包含若干特征列和一个标签列（默认列名 `label`）。
将训练集、验证集、测试集分别放置于 `data/train.csv`、`data/val.csv`、`data/test.csv`。

## 训练

```bash
python scripts/train.py --config configs/config.yaml
```

训练过程中，最优模型权重将保存至 `checkpoints/best_model.pt`。

## 推理

```bash
python scripts/inference.py \
    --config configs/config.yaml \
    --checkpoint checkpoints/best_model.pt \
    --input data/test.csv
```

推理结果将保存至输入文件同目录下的 `*_predictions.csv` 文件中。

## 模型架构

```
输入特征 (B, S, input_dim)
    │
    ▼
线性投影 + 位置编码
    │
    ▼
Transformer 编码器 (多头注意力 × N 层)
    │
    ▼
全局平均池化
    │
    ▼
全连接分类头
    │
    ▼
分类输出 (B, num_classes)
```
