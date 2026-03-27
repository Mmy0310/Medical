"""
推理脚本：使用训练好的模型对新医学案例进行预测。

用法：
    python scripts/inference.py --config configs/config.yaml --checkpoint checkpoints/best_model.pt --input data/test.csv
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from models.model import MedicalTransformerFC
from utils.utils import load_config, get_device, load_checkpoint, setup_logging


def inference(config_path: str, checkpoint_path: str, input_path: str) -> None:
    cfg = load_config(config_path)
    setup_logging()

    device = get_device(cfg["training"]["device"])

    model = MedicalTransformerFC(
        input_dim=cfg["model"]["input_dim"],
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    load_checkpoint(model, checkpoint_path)
    model.eval()

    df = pd.read_csv(input_path)
    label_col = cfg["data"]["label_col"]
    feature_cols = cfg["data"].get("feature_cols")
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != label_col]

    features = df[feature_cols].values.astype(np.float32)

    scaler_path = os.path.splitext(checkpoint_path)[0] + "_scaler.pkl"
    if os.path.exists(scaler_path):
        import pickle
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        features = scaler.transform(features)
        logging.info(f"已从 {scaler_path} 加载训练集标准化参数")
    else:
        logging.warning(
            f"未找到 scaler 文件 {scaler_path}，将在测试集上拟合新的标准化器（仅供调试）。"
        )
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

    n_samples, n_total = features.shape
    input_dim = cfg["model"]["input_dim"]
    seq_len = n_total // input_dim
    features = features.reshape(n_samples, seq_len, input_dim)
    features_tensor = torch.tensor(features, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(features_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

    df["predicted_label"] = preds
    for i in range(probs.shape[1]):
        df[f"prob_class_{i}"] = probs[:, i]

    output_path = os.path.splitext(input_path)[0] + "_predictions.csv"
    df.to_csv(output_path, index=False)
    logging.info(f"预测结果已保存至 {output_path}")


def main():
    parser = argparse.ArgumentParser(description="医学案例推理脚本")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="配置文件路径"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--input", type=str, required=True, help="输入 CSV 数据路径")
    args = parser.parse_args()
    inference(args.config, args.checkpoint, args.input)


if __name__ == "__main__":
    main()
