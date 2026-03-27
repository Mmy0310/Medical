import os
import pickle
import random
import logging
from typing import Dict

import numpy as np
import torch
import yaml


def load_config(config_path: str) -> Dict:
    """从 YAML 文件加载配置。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """固定随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str = "cuda") -> torch.device:
    """返回可用的计算设备。"""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    path: str,
    epoch: int,
    metrics: Dict,
    scaler=None,
) -> None:
    """保存模型检查点；如提供 scaler 则同时保存标准化参数。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "metrics": metrics}, path)
    logging.info(f"检查点已保存至 {path}")
    if scaler is not None:
        scaler_path = os.path.splitext(path)[0] + "_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        logging.info(f"标准化参数已保存至 {scaler_path}")


def load_checkpoint(model: torch.nn.Module, path: str) -> Dict:
    """从检查点文件加载模型权重。"""
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"已从 {path} 加载检查点（epoch {checkpoint['epoch']}）")
    return checkpoint


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """配置日志输出到控制台和文件。"""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, "train.log"), encoding="utf-8"),
        ],
    )
