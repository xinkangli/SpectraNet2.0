"""
data_utils.py
─────────────
统一数据加载工具。

文件格式约定（两个文件相同）：
  - 第一行：列名（表头）
  - 最后一列：y（标签，字符串或数字均可）
  - 其余列：x（光谱特征值）
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d


# ──────────────────────────────────────────────────────────────
#  读取 Excel
# ──────────────────────────────────────────────────────────────

def load_excel(path: str, target_len: int = None):
    """
    读取 xlsx 文件。
    返回:
        X           : np.float32  (N, signal_len)   样本级归一化到 [-1,1]
        y           : np.int64    (N,)
        class_names : list[str]
        signal_len  : int
    """
    df = pd.read_excel(path, header=0)
    print(f"  读取 {path}: {df.shape[0]} 行 x {df.shape[1]} 列")

    y_raw = df.iloc[:, -1].values
    X_raw = df.iloc[:, :-1].values.astype(np.float32)

    le = LabelEncoder()
    y  = le.fit_transform(y_raw).astype(np.int64)

    if target_len is not None and X_raw.shape[1] != target_len:
        X_raw = _resize(X_raw, target_len)

    signal_len = X_raw.shape[1]
    X = _normalize(X_raw)

    return X, y, list(le.classes_), signal_len


def split_pretrain_newbatch(X, y, new_batch_ratio=0.2, seed=42):
    """
    当没有独立新批次文件时，从主数据集内部切分。
    前 (1-ratio) 作预训练，后 ratio 模拟新批次。
    """
    idx = np.arange(len(X))
    pretrain_idx, new_idx = train_test_split(
        idx, test_size=new_batch_ratio, stratify=y, random_state=seed
    )
    return X[pretrain_idx], y[pretrain_idx], X[new_idx], y[new_idx]


# ──────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────

class SpectralDataset(Dataset):
    """有标签数据集（Stage1 监督训练 / Stage3 few-shot）"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class UnlabeledDataset(Dataset):
    """无标签数据集（Stage1 自监督 / Stage2 自适应）"""
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (N,1,L)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i],)


# ──────────────────────────────────────────────────────────────
#  Few-shot 采样
# ──────────────────────────────────────────────────────────────

def sample_fewshot(X, y, shots_per_class: int, seed: int = 0):
    """
    每类随机抽 shots_per_class 条作支持集，其余作查询集。
    """
    rng     = np.random.default_rng(seed)
    classes = np.unique(y)
    sup_idx, qry_idx = [], []
    for c in classes:
        cidx   = np.where(y == c)[0]
        k      = min(shots_per_class, len(cidx))
        chosen = rng.choice(cidx, k, replace=False)
        sup_idx.extend(chosen.tolist())
        qry_idx.extend(np.setdiff1d(cidx, chosen).tolist())
    return (X[sup_idx], y[sup_idx],
            X[qry_idx], y[qry_idx])


# ──────────────────────────────────────────────────────────────
#  内部工具
# ──────────────────────────────────────────────────────────────

def _resize(X, target_len):
    old_x = np.linspace(0, 1, X.shape[1])
    new_x = np.linspace(0, 1, target_len)
    return np.stack(
        [interp1d(old_x, row)(new_x) for row in X]
    ).astype(np.float32)


def _normalize(X):
    """样本级 min-max 归一化到 [-1, 1]"""
    mn = X.min(axis=1, keepdims=True)
    mx = X.max(axis=1, keepdims=True)
    d  = np.where((mx - mn) < 1e-8, 1.0, mx - mn)
    return (2 * (X - mn) / d - 1).astype(np.float32)
