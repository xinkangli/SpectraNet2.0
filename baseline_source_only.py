"""
baseline_source_only.py
────────────────────────────────────────────────────────────────
对比实验 Step 1：Source-only（源域训练 → 直接测试目标域）

流程：
  1. 在源数据（--data）上有监督训练 Encoder + 分类头
  2. 直接在目标数据（--new_batch）上推理，不做任何适应
  3. 输出 acc / auc / sp / se / F1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 用法示例
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 基本运行
python baseline_source_only.py \
    --data data.xlsx \
    --new_batch dataceshi1600.xlsx

# 自定义训练参数
python baseline_source_only.py \
    --data data.xlsx \
    --new_batch dataceshi1600.xlsx \
    --epochs 80 --lr 1e-3 --batch 64 \
    --save_dir checkpoints_source_only

# 批量写入结果文件（与 pipeline1.py 保持一致）
echo "===== Source-only =====" >> results_all.txt
python baseline_source_only.py \
    --data data.xlsx \
    --new_batch dataceshi1600.xlsx >> results_all.txt 2>&1
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    f1_score, classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import label_binarize

from data_utils import load_excel, SpectralDataset
from models import Encoder1D, ClassificationHead


# ──────────────────────────────────────────────────────────────
#  Encoder 超参（与 pipeline1.py 保持一致）
# ──────────────────────────────────────────────────────────────
ENCODER_OUT_DIM = 128
ENCODER_HIDDEN  = 256
TARGET_LEN      = 256


# ──────────────────────────────────────────────────────────────
#  命令行参数
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="对比实验 Step1：Source-only 基线"
    )
    p.add_argument("--data",      required=True, metavar="FILE",
                   help="源域数据文件（如 data.xlsx）")
    p.add_argument("--new_batch", required=True, metavar="FILE",
                   help="目标域数据文件（如 dataceshi1600.xlsx）")
    p.add_argument("--save_dir",  default="checkpoints_source_only", metavar="DIR",
                   help="权重保存目录（默认 checkpoints_source_only/）")
    p.add_argument("--epochs",    type=int,   default=60,   help="训练轮数（默认60）")
    p.add_argument("--lr",        type=float, default=1e-3, help="学习率（默认1e-3）")
    p.add_argument("--batch",     type=int,   default=64,   help="batch size（默认64）")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
#  训练：源域有监督
# ──────────────────────────────────────────────────────────────

def train_source(encoder, head, X, y, epochs, lr, batch_size, device, save_path):
    ds      = SpectralDataset(X, y)
    loader  = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    optim   = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=lr, weight_decay=1e-4
    )
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    print(f"\n  [Source Training] samples={len(X)}  epochs={epochs}  "
          f"lr={lr}  batch={batch_size}")

    for ep in range(1, epochs + 1):
        encoder.train(); head.train()
        total_loss, correct, total = 0., 0, 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            z      = encoder(xb)
            logits = head(z)
            loss   = loss_fn(logits, yb)

            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)

        sched.step()
        avg_loss = total_loss / total
        acc      = correct / total

        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"train_acc={acc:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "encoder": encoder.state_dict(),
                "head":    head.state_dict(),
            }, save_path)

    print(f"  ✓ 最优模型已保存 → {save_path}")


# ──────────────────────────────────────────────────────────────
#  评估：直接在目标域推理
# ──────────────────────────────────────────────────────────────

def evaluate_on_target(encoder, head, X_tgt, y_tgt, n_classes, device):
    encoder.eval(); head.eval()
    with torch.no_grad():
        xb    = torch.tensor(X_tgt, dtype=torch.float32).unsqueeze(1).to(device)
        z     = encoder(xb)
        logits = head(z)
        probs  = F.softmax(logits, dim=-1).cpu().numpy()
    preds  = probs.argmax(1)
    y_true = np.array(y_tgt)

    acc = accuracy_score(y_true, preds)

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=np.arange(n_classes))
            auc   = roc_auc_score(y_bin, probs,
                                  multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_true, preds, labels=np.arange(n_classes))
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
    f1 = f1_score(y_true, preds, average="macro", zero_division=0)

    print(f"\n{'='*65}")
    print(f"  [Source-only 结果] 直接测试目标域（无任何适应）")
    print(f"{'='*65}")
    print(f"  acc={acc:.4f}  auc={auc:.4f}  "
          f"sp={sp:.4f}  se={se:.4f}  F1={f1:.4f}")
    print(f"{'─'*65}")
    print(classification_report(y_true, preds, zero_division=0))
    print(f"{'='*65}\n")

    return dict(acc=acc, auc=auc, sp=sp, se=se, f1=f1)


# ──────────────────────────────────────────────────────────────
#  main
# ──────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tgt_len = TARGET_LEN if TARGET_LEN > 0 else None

    print(f"\n{'='*65}")
    print(f"  对比实验 Step 1：Source-only 基线")
    print(f"  Device   : {device}")
    print(f"  save_dir : {Path(args.save_dir).resolve()}")
    print(f"{'='*65}")

    # ── 源域数据 ─────────────────────────────────────────────
    if not Path(args.data).exists():
        print(f"  ✗ 找不到源域文件: {args.data}"); sys.exit(1)
    X_src, y_src, class_names, signal_len = load_excel(args.data, tgt_len)
    n_classes = len(class_names)
    print(f"\n[源域] {args.data}")
    print(f"  samples={len(X_src)}  signal_len={signal_len}  "
          f"n_classes={n_classes}  classes={class_names}")

    # ── 目标域数据 ────────────────────────────────────────────
    if not Path(args.new_batch).exists():
        print(f"  ✗ 找不到目标域文件: {args.new_batch}"); sys.exit(1)
    X_tgt, y_tgt, tgt_classes, _ = load_excel(args.new_batch, tgt_len)
    print(f"\n[目标域] {args.new_batch}")
    print(f"  samples={len(X_tgt)}  classes={tgt_classes}")

    # ── 构建模型 ──────────────────────────────────────────────
    encoder = Encoder1D(
        signal_len=signal_len,
        out_dim=ENCODER_OUT_DIM,
        hidden_dim=ENCODER_HIDDEN,
    ).to(device)
    head = ClassificationHead(ENCODER_OUT_DIM, n_classes).to(device)
    print(f"\n  Encoder 参数量: {sum(p.numel() for p in encoder.parameters()):,}")

    # ── Step 1：源域训练 ──────────────────────────────────────
    save_path = Path(args.save_dir) / "source_only.pt"
    train_source(encoder, head, X_src, y_src,
                 args.epochs, args.lr, args.batch, device, str(save_path))

    # ── Step 2：目标域直接评估（不适应）──────────────────────
    ckpt = torch.load(str(save_path), map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])

    evaluate_on_target(encoder, head, X_tgt, y_tgt, n_classes, device)


if __name__ == "__main__":
    main()
