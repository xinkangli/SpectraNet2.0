"""
stage3_fewshot.py
─────────────────
Stage 3：Few-Shot 分类。

原有三种（保留对比基线）：
  linear          → 冻结 encoder，AdamW 线性分类头
  finetune        → 解冻最后 2 层微调（少样本易过拟合，仅作对比）
  prototype       → 最近类原型，无需训练

新增四种（均冻结 encoder，围绕"保住表征"优化）：
  linear_weighted → 线性头 + 类别反频率加权损失（解决类别不平衡）
  linear_lbfgs    → L-BFGS 直接求解线性头（少样本收敛更稳）
  linear_norm     → 余弦分类器：L2 归一化特征 + 归一化权重
  svm             → 冻结特征 + RBF-SVM（少样本经典强基线）

每种方法统一输出：acc / auc / sp(特异度) / se(敏感度) / F1
最后打印汇总排名表。

mode="all" → 全部七种
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    f1_score, classification_report,
    confusion_matrix,
)
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize

from models import ClassificationHead
from data_utils import SpectralDataset, sample_fewshot


# ──────────────────────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────────────────────

def run_stage3(encoder, X_new, y_new, n_classes,
               shots_per_class=5,
               mode="all",
               encoder_dim=128,
               epochs=60, lr=1e-3,
               device="cpu",
               save_dir="checkpoints"):

    print(f"  [Stage3] mode={mode}  shots={shots_per_class}/class  n_classes={n_classes}")

    X_sup, y_sup, X_qry, y_qry = sample_fewshot(X_new, y_new, shots_per_class)
    print(f"  支持集={len(y_sup)}  查询集={len(y_qry)}")

    # 每种方法的调用表
    run_map = {
        "linear": lambda: _linear(
            encoder, X_sup, y_sup, X_qry, y_qry,
            n_classes, encoder_dim, epochs, lr, device),
        "finetune": lambda: _finetune(
            encoder, X_sup, y_sup, X_qry, y_qry,
            n_classes, encoder_dim, epochs, lr, device),
        "prototype": lambda: _prototype(
            encoder, X_sup, y_sup, X_qry, y_qry,
            n_classes, device),
        "linear_weighted": lambda: _linear_weighted(
            encoder, X_sup, y_sup, X_qry, y_qry,
            n_classes, encoder_dim, epochs, lr, device),
        "linear_lbfgs": lambda: _linear_lbfgs(
            encoder, X_sup, y_sup, X_qry, y_qry,
            n_classes, encoder_dim, device),
        "linear_norm": lambda: _linear_norm(
            encoder, X_sup, y_sup, X_qry, y_qry,
            n_classes, encoder_dim, epochs, lr, device),
        "svm": lambda: _svm(
            encoder, X_sup, y_sup, X_qry, y_qry,
            n_classes, device),
    }

    keys = list(run_map.keys()) if mode == "all" else [mode]
    if mode != "all" and mode not in run_map:
        raise ValueError(f"未知 mode={mode}，可选: {list(run_map.keys())} / all")

    results = {}
    for k in keys:
        print(f"\n{'─'*55}")
        print(f"  ▶  {k}")
        print(f"{'─'*55}")
        results[k] = run_map[k]()

    _print_summary(results)


# ──────────────────────────────────────────────────────────────
#  统一指标计算
# ──────────────────────────────────────────────────────────────

def _compute_metrics(y_true, y_pred, y_prob, n_classes, name):
    """
    计算 acc / auc / sp / se / F1，打印详细报告，返回 dict。
    sp = macro 平均特异度（TN/(TN+FP)）
    se = macro 平均敏感度/召回率（TP/(TP+FN)）
    """
    acc = accuracy_score(y_true, y_pred)

    # AUC
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=np.arange(n_classes))
            auc   = roc_auc_score(y_bin, y_prob,
                                  multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    # se 和 sp（逐类从混淆矩阵算，再 macro 平均）
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_classes))
    se_list, sp_list = [], []
    for i in range(n_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        se_list.append(tp / (tp + fn + 1e-8))
        sp_list.append(tn / (tn + fp + 1e-8))
    se = float(np.mean(se_list))
    sp = float(np.mean(sp_list))

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"\n  ┌─ {name} ─────────────────────────────")
    print(f"  │  acc={acc:.4f}  auc={auc:.4f}  "
          f"sp={sp:.4f}  se={se:.4f}  F1={f1:.4f}")
    print(f"  └{'─'*45}")
    print(classification_report(y_true, y_pred, zero_division=0))

    return dict(acc=acc, auc=auc, sp=sp, se=se, f1=f1)


def _extract_z(encoder, X, device):
    """提取 encoder 特征 → numpy (N, D)"""
    encoder.eval()
    with torch.no_grad():
        z = encoder(
            torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        )
    return z.cpu().numpy()


def _head_probs(encoder, head, X, device):
    """线性头 softmax 概率 → numpy (N, C)"""
    encoder.eval(); head.eval()
    with torch.no_grad():
        z = encoder(
            torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(device)
        )
        probs = F.softmax(head(z), dim=-1).cpu().numpy()
    return probs


# ──────────────────────────────────────────────────────────────
#  (1) Linear Probe（AdamW，基线）
# ──────────────────────────────────────────────────────────────

def _linear(encoder, X_sup, y_sup, X_qry, y_qry,
            n_classes, encoder_dim, epochs, lr, device):
    encoder.freeze()
    head, opt, sch = _build_head(encoder_dim, n_classes, lr, epochs, device)
    _fit_head(encoder, head, opt, sch, X_sup, y_sup, epochs, device,
              loss_fn=F.cross_entropy)
    probs = _head_probs(encoder, head, X_qry, device)
    m = _compute_metrics(y_qry, probs.argmax(1), probs, n_classes,
                         "Linear Probe (AdamW)")
    encoder.unfreeze()
    return m


# ──────────────────────────────────────────────────────────────
#  (2) Finetune（解冻最后 2 层，对比用）
# ──────────────────────────────────────────────────────────────

def _finetune(encoder, X_sup, y_sup, X_qry, y_qry,
              n_classes, encoder_dim, epochs, lr, device):
    encoder.unfreeze(last_n_layers=2)
    head   = ClassificationHead(encoder_dim, n_classes).to(device)
    params = [p for p in encoder.parameters() if p.requires_grad] + \
             list(head.parameters())
    opt  = torch.optim.AdamW(params, lr=lr * 0.3, weight_decay=1e-3)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = _make_loader(X_sup, y_sup)

    for _ in range(epochs):
        encoder.train(); head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(head(encoder(xb)), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

    probs = _head_probs(encoder, head, X_qry, device)
    m = _compute_metrics(y_qry, probs.argmax(1), probs, n_classes,
                         "Finetune (last 2 layers)")
    encoder.unfreeze()
    return m


# ──────────────────────────────────────────────────────────────
#  (3) Prototype Network（对比用）
# ──────────────────────────────────────────────────────────────

def _prototype(encoder, X_sup, y_sup, X_qry, y_qry, n_classes, device):
    z_sup = torch.tensor(_extract_z(encoder, X_sup, device))
    z_qry = torch.tensor(_extract_z(encoder, X_qry, device))

    protos = torch.stack([
        z_sup[torch.tensor(y_sup) == c].mean(0)
        for c in range(n_classes)
    ])  # (C, D)

    dists = torch.cdist(z_qry, protos)                   # (N, C)
    probs = F.softmax(-dists, dim=-1).numpy()
    preds = dists.argmin(1).numpy()
    return _compute_metrics(y_qry, preds, probs, n_classes,
                            "Prototype Network")


# ──────────────────────────────────────────────────────────────
#  (4) Linear + 类别加权损失（解决类别不平衡）
# ──────────────────────────────────────────────────────────────

def _linear_weighted(encoder, X_sup, y_sup, X_qry, y_qry,
                     n_classes, encoder_dim, epochs, lr, device):
    encoder.freeze()

    counts  = np.bincount(y_sup, minlength=n_classes).astype(float)
    counts  = np.where(counts == 0, 1.0, counts)
    w       = torch.tensor(1.0 / counts, dtype=torch.float32).to(device)
    w       = w / w.sum() * n_classes   # 均值归一化到 1

    def weighted_ce(logits, labels):
        return F.cross_entropy(logits, labels, weight=w)

    head, opt, sch = _build_head(encoder_dim, n_classes, lr, epochs, device)
    _fit_head(encoder, head, opt, sch, X_sup, y_sup, epochs, device,
              loss_fn=weighted_ce)
    probs = _head_probs(encoder, head, X_qry, device)
    m = _compute_metrics(y_qry, probs.argmax(1), probs, n_classes,
                         "Linear + Weighted Loss (class-balanced)")
    encoder.unfreeze()
    return m


# ──────────────────────────────────────────────────────────────
#  (5) Linear L-BFGS（少样本收敛最稳）
# ──────────────────────────────────────────────────────────────

def _linear_lbfgs(encoder, X_sup, y_sup, X_qry, y_qry,
                  n_classes, encoder_dim, device, max_iter=500):
    """
    提取全部特征后用 L-BFGS 直接求解线性头。
    少样本时比 SGD/Adam 收敛稳定，不受学习率影响。
    """
    encoder.freeze()

    z_sup   = torch.tensor(_extract_z(encoder, X_sup, device), dtype=torch.float32)
    z_qry_np = _extract_z(encoder, X_qry, device)
    y_t     = torch.tensor(y_sup, dtype=torch.long)

    head = nn.Linear(encoder_dim, n_classes)
    opt  = torch.optim.LBFGS(head.parameters(), lr=0.1,
                              max_iter=max_iter,
                              line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        F.cross_entropy(head(z_sup), y_t).backward()
        return F.cross_entropy(head(z_sup), y_t)

    opt.step(closure)

    with torch.no_grad():
        probs = F.softmax(
            head(torch.tensor(z_qry_np, dtype=torch.float32)), dim=-1
        ).numpy()
    m = _compute_metrics(y_qry, probs.argmax(1), probs, n_classes,
                         "Linear L-BFGS (direct solver)")
    encoder.unfreeze()
    return m


# ──────────────────────────────────────────────────────────────
#  (6) Cosine 分类器（L2 归一化特征 + 归一化权重）
# ──────────────────────────────────────────────────────────────

def _linear_norm(encoder, X_sup, y_sup, X_qry, y_qry,
                 n_classes, encoder_dim, epochs, lr, device):
    """
    特征和分类权重都做 L2 归一化，等效于余弦相似度分类。
    少样本时对特征尺度变化更鲁棒，比普通线性头更稳定。
    """
    encoder.freeze()

    class CosineHead(nn.Module):
        def __init__(self, d, c, scale=10.0):
            super().__init__()
            self.w     = nn.Parameter(torch.randn(c, d))
            self.scale = scale
        def forward(self, z):
            return self.scale * (
                F.normalize(z, dim=-1) @ F.normalize(self.w, dim=-1).T
            )

    head   = CosineHead(encoder_dim, n_classes).to(device)
    opt    = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = _make_loader(X_sup, y_sup)

    for _ in range(epochs):
        encoder.eval(); head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                z = encoder(xb)
            loss = F.cross_entropy(head(z), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()

    encoder.eval(); head.eval()
    with torch.no_grad():
        z_q = encoder(
            torch.tensor(X_qry, dtype=torch.float32).unsqueeze(1).to(device)
        )
        probs = F.softmax(head(z_q), dim=-1).cpu().numpy()

    m = _compute_metrics(y_qry, probs.argmax(1), probs, n_classes,
                         "Cosine Classifier (L2-normalized features)")
    encoder.unfreeze()
    return m


# ──────────────────────────────────────────────────────────────
#  (7) SVM（冻结特征 + RBF-SVM，经典少样本强基线）
# ──────────────────────────────────────────────────────────────

def _svm(encoder, X_sup, y_sup, X_qry, y_qry, n_classes, device):
    """
    encoder 特征 + RBF-SVM。
    class_weight="balanced" 自动处理不平衡。
    少样本高维特征场景通常是最强基线之一。
    """
    z_sup = _extract_z(encoder, X_sup, device)
    z_qry = _extract_z(encoder, X_qry, device)

    clf = SVC(kernel="rbf", C=10.0, gamma="scale",
              probability=True, class_weight="balanced")
    clf.fit(z_sup, y_sup)

    probs = clf.predict_proba(z_qry)
    preds = clf.predict(z_qry)
    return _compute_metrics(y_qry, preds, probs, n_classes,
                            "SVM (RBF, class-balanced)")


# ──────────────────────────────────────────────────────────────
#  汇总排名表
# ──────────────────────────────────────────────────────────────

def _print_summary(results: dict):
    print(f"\n{'='*72}")
    print(f"  {'方法':<30} {'acc':>7} {'auc':>7} {'sp':>7} {'se':>7} {'F1':>7}")
    print(f"  {'─'*66}")
    for name, m in sorted(results.items(), key=lambda x: -x[1]["acc"]):
        tag = " ◀ best" if name == max(results, key=lambda k: results[k]["acc"]) else ""
        print(f"  {name:<30} "
              f"{m['acc']:>7.4f} {m['auc']:>7.4f} "
              f"{m['sp']:>7.4f} {m['se']:>7.4f} {m['f1']:>7.4f}{tag}")
    print(f"{'='*72}\n")


# ──────────────────────────────────────────────────────────────
#  内部工具
# ──────────────────────────────────────────────────────────────

def _make_loader(X, y):
    return DataLoader(SpectralDataset(X, y),
                      batch_size=min(32, len(y)),
                      shuffle=True, drop_last=False)


def _build_head(encoder_dim, n_classes, lr, epochs, device):
    head = ClassificationHead(encoder_dim, n_classes).to(device)
    opt  = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-3)
    sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    return head, opt, sch


def _fit_head(encoder, head, opt, sch, X_sup, y_sup,
              epochs, device, loss_fn):
    loader = _make_loader(X_sup, y_sup)
    for _ in range(epochs):
        encoder.eval(); head.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.no_grad():
                z = encoder(xb)
            loss = loss_fn(head(z), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sch.step()