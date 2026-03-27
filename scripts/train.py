"""
训练脚本：使用 Transformer + 全连接网络对医学案例进行学习。

用法：
    python scripts/train.py --config configs/config.yaml
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score

from data.dataset import build_dataloaders
from models.model import MedicalTransformerFC
from utils.utils import load_config, set_seed, get_device, save_checkpoint, setup_logging


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    n = len(all_labels)
    avg_loss = total_loss / n if n > 0 else 0.0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    setup_logging()
    set_seed(cfg["training"]["seed"])

    device = get_device(cfg["training"]["device"])
    logging.info(f"使用设备: {device}")

    train_loader, val_loader, _, scaler = build_dataloaders(
        train_path=cfg["data"]["train_path"],
        val_path=cfg["data"]["val_path"],
        test_path=cfg["data"]["test_path"],
        label_col=cfg["data"]["label_col"],
        feature_cols=cfg["data"].get("feature_cols"),
        batch_size=cfg["training"]["batch_size"],
    )

    model = MedicalTransformerFC(
        input_dim=cfg["model"]["input_dim"],
        d_model=cfg["model"]["d_model"],
        nhead=cfg["model"]["nhead"],
        num_encoder_layers=cfg["model"]["num_encoder_layers"],
        dim_feedforward=cfg["model"]["dim_feedforward"],
        dropout=cfg["model"]["dropout"],
        num_classes=cfg["model"]["num_classes"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    best_val_f1 = 0.0
    save_dir = cfg["training"]["save_dir"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            if (batch_idx + 1) % cfg["training"]["log_interval"] == 0:
                avg = running_loss / cfg["training"]["log_interval"]
                logging.info(f"Epoch {epoch} | Batch {batch_idx + 1} | Loss: {avg:.4f}")
                running_loss = 0.0

        scheduler.step()

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        logging.info(
            f"[Epoch {epoch}] Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_checkpoint(
                model,
                os.path.join(save_dir, "best_model.pt"),
                epoch,
                {"val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1},
                scaler=scaler,
            )

    logging.info(f"训练完成。最佳验证 F1: {best_val_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="医学案例 Transformer 训练脚本")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="配置文件路径"
    )
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
