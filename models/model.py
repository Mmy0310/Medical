import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """正弦位置编码，用于为序列中的每个位置添加位置信息。"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class MedicalTransformerFC(nn.Module):
    """
    医学案例学习推理模型。

    结合 Transformer 编码器与全连接神经网络对医学特征进行分类推理。
    输入特征经过线性投影嵌入，加上位置编码后送入 Transformer 编码器，
    再经过若干全连接层完成分类。
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        """
        Args:
            input_dim: 输入特征维度（每个时间步的特征数）。
            d_model: Transformer 模型维度。
            nhead: 多头注意力头数。
            num_encoder_layers: Transformer 编码器层数。
            dim_feedforward: Transformer 前馈网络维度。
            dropout: Dropout 概率。
            num_classes: 分类类别数。
        """
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, input_dim)。
            src_key_padding_mask: 填充掩码，形状为 (batch_size, seq_len)，
                                  True 表示对应位置为填充。
        Returns:
            logits: 分类输出，形状为 (batch_size, num_classes)。
        """
        x = self.input_projection(x)          # (B, S, d_model)
        x = self.positional_encoding(x)        # (B, S, d_model)
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )                                      # (B, S, d_model)
        x = x.mean(dim=1)                      # 全局平均池化 (B, d_model)
        logits = self.classifier(x)            # (B, num_classes)
        return logits
