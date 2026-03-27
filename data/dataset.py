import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple


class MedicalDataset(Dataset):
    """
    医学案例数据集。

    从 CSV 文件中读取特征和标签，支持标准化处理。
    特征列可以包含多个时间步（宽格式），每行代表一个样本。
    """

    def __init__(
        self,
        csv_path: str,
        label_col: str = "label",
        feature_cols: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
        seq_len: int = 1,
    ):
        """
        Args:
            csv_path: CSV 数据文件路径。
            label_col: 标签列名。
            feature_cols: 特征列名列表；为 None 时使用除标签列外的所有列。
            scaler: 已拟合的 StandardScaler；为 None 时在训练集上拟合。
            seq_len: 将特征重塑为 (seq_len, input_dim) 时使用的序列长度。
        """
        df = pd.read_csv(csv_path)

        self.labels = torch.tensor(df[label_col].values, dtype=torch.long)

        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != label_col]
        features = df[feature_cols].values.astype(np.float32)

        if scaler is None:
            scaler = StandardScaler()
            features = scaler.fit_transform(features)
        else:
            features = scaler.transform(features)
        self.scaler = scaler

        # 重塑为 (N, seq_len, input_dim)
        n_samples, n_total = features.shape
        if n_total % seq_len != 0:
            raise ValueError(
                f"特征数量 {n_total} 不能被 seq_len={seq_len} 整除。"
                f"请调整 config.yaml 中的 input_dim 参数使其能够整除特征总数，"
                f"或对特征进行填充。"
            )
        input_dim = n_total // seq_len
        self.features = torch.tensor(
            features.reshape(n_samples, seq_len, input_dim), dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def build_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    label_col: str = "label",
    feature_cols: Optional[List[str]] = None,
    batch_size: int = 32,
    seq_len: int = 1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
    """
    构建训练、验证和测试数据加载器。

    Returns:
        (train_loader, val_loader, test_loader, scaler)
    """
    train_ds = MedicalDataset(
        train_path, label_col=label_col, feature_cols=feature_cols, seq_len=seq_len
    )
    scaler = train_ds.scaler

    val_ds = MedicalDataset(
        val_path,
        label_col=label_col,
        feature_cols=feature_cols,
        scaler=scaler,
        seq_len=seq_len,
    )
    test_ds = MedicalDataset(
        test_path,
        label_col=label_col,
        feature_cols=feature_cols,
        scaler=scaler,
        seq_len=seq_len,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader, scaler
