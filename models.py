"""
models.py
─────────
所有模型组件，三个阶段共用同一个 Encoder1D。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────
#  核心 Encoder
# ──────────────────────────────────────────────────────────────

class Encoder1D(nn.Module):
    """
    1D-CNN Encoder
    输入:  (B, 1, L)
    输出:  (B, out_dim)

    backbone: 4 个 Conv 块（逐步降采样）→ GAP → FC → LayerNorm
    """
    def __init__(self, signal_len: int, out_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.out_dim    = out_dim
        self.hidden_dim = hidden_dim

        def conv_block(in_c, out_c, k, stride, pad):
            return nn.Sequential(
                nn.Conv1d(in_c, out_c, k, stride=stride, padding=pad, bias=False),
                nn.BatchNorm1d(out_c),
                nn.GELU(),
            )

        self.backbone = nn.Sequential(
            conv_block(1,              hidden_dim // 4, k=15, stride=2, pad=7),
            conv_block(hidden_dim // 4, hidden_dim // 2, k=7,  stride=2, pad=3),
            conv_block(hidden_dim // 2, hidden_dim,      k=5,  stride=2, pad=2),
            conv_block(hidden_dim,      hidden_dim,      k=3,  stride=1, pad=1),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)   # 输出与 signal_len 无关
        self.fc  = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):          # x: (B,1,L)
        h = self.backbone(x)       # (B, hidden_dim, L')
        h = self.gap(h).squeeze(-1)  # (B, hidden_dim)
        return self.fc(h)          # (B, out_dim)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self, last_n_layers: int = None):
        """
        last_n_layers=None  → 解冻全部
        last_n_layers=N     → 只解冻最后 N 个子模块
        """
        for p in self.parameters():
            p.requires_grad = True
        if last_n_layers is not None:
            for p in self.parameters():
                p.requires_grad = False
            mods = list(self.backbone) + list(self.fc)
            for m in mods[-last_n_layers:]:
                for p in m.parameters():
                    p.requires_grad = True


# ──────────────────────────────────────────────────────────────
#  辅助头
# ──────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    """SimCLR / BYOL 投影头，输出 L2 归一化特征"""
    def __init__(self, in_dim: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, z):
        return F.normalize(self.net(z), dim=-1)


class PredictionHead(nn.Module):
    """BYOL online 预测头"""
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, z):
        return self.net(z)


class DecoderHead(nn.Module):
    """AutoEncoder / Masked Reconstruction 解码器（转置卷积上采样）"""
    def __init__(self, in_dim: int, signal_len: int, hidden_dim: int = 256):
        super().__init__()
        self.signal_len = signal_len
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(in_dim, hidden_dim * 8)

        def deconv_block(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(out_c),
                nn.GELU(),
            )

        self.upsampler = nn.Sequential(
            deconv_block(hidden_dim,      hidden_dim // 2),
            deconv_block(hidden_dim // 2, hidden_dim // 4),
            deconv_block(hidden_dim // 4, hidden_dim // 8),
            nn.ConvTranspose1d(hidden_dim // 8, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        h     = self.fc(z).view(z.size(0), self.hidden_dim, 8)
        x_hat = self.upsampler(h)
        # 插值到原始长度，消除上采样误差
        x_hat = F.interpolate(x_hat, size=self.signal_len, mode='linear', align_corners=False)
        return x_hat   # (B, 1, L)


class ClassificationHead(nn.Module):
    """Stage3 线性分类头"""
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, z):
        return self.fc(z)
