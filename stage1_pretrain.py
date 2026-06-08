"""
stage1_pretrain.py
──────────────────
Stage 1：通用预训练
  mode="supervised"  → 有监督交叉熵预训练
  mode="autoencoder" → 自监督 AutoEncoder 预训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import ClassificationHead, DecoderHead
from data_utils import SpectralDataset, UnlabeledDataset


# ──────────────────────────────────────────────────────────────

def run_stage1(encoder, X, y, n_classes, signal_len,
               mode="supervised",
               epochs=50, lr=1e-3, batch_size=64,
               device="cpu", save_path="checkpoints/stage1_encoder.pt"):

    print(f"  [Stage1] mode={mode}  samples={len(X)}  epochs={epochs}")
    encoder = encoder.to(device)

    if mode == "supervised":
        _train_supervised(encoder, X, y, n_classes,
                          epochs, lr, batch_size, device, save_path)
    elif mode == "autoencoder":
        _train_autoencoder(encoder, X, signal_len,
                           epochs, lr, batch_size, device, save_path)
    else:
        raise ValueError(f"未知 stage1 mode: {mode}")


# ──────────────────────────────────────────────────────────────
#  监督预训练
# ──────────────────────────────────────────────────────────────

def _train_supervised(encoder, X, y, n_classes,
                      epochs, lr, batch_size, device, save_path):
    ds     = SpectralDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    head   = ClassificationHead(encoder.out_dim, n_classes).to(device)
    optim  = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4
    )
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    for ep in range(1, epochs + 1):
        encoder.train(); head.train()
        total_loss, correct, total = 0., 0, 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            z    = encoder(xb)
            logits = head(z)
            loss = loss_fn(logits, yb)

            optim.zero_grad(); loss.backward(); optim.step()

            total_loss += loss.item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)

        sched.step()
        avg_loss = total_loss / total
        acc      = correct / total

        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:3d}/{epochs}  loss={avg_loss:.4f}  acc={acc:.3f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(encoder.state_dict(), save_path)

    print(f"  ✓ Stage1 最优模型已保存 → {save_path}")


# ──────────────────────────────────────────────────────────────
#  自监督 AutoEncoder 预训练
# ──────────────────────────────────────────────────────────────

def _train_autoencoder(encoder, X, signal_len,
                       epochs, lr, batch_size, device, save_path):
    ds      = UnlabeledDataset(X)
    loader  = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    decoder = DecoderHead(encoder.out_dim, signal_len, hidden_dim=256).to(device)

    optim   = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=1e-4
    )
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

    best_loss = float("inf")
    for ep in range(1, epochs + 1):
        encoder.train(); decoder.train()
        total_loss, total = 0., 0

        for xb, in loader:
            xb = xb.to(device)
            z     = encoder(xb)
            x_hat = decoder(z)
            loss  = F.mse_loss(x_hat, xb)

            optim.zero_grad(); loss.backward(); optim.step()
            total_loss += loss.item() * len(xb)
            total      += len(xb)

        sched.step()
        avg = total_loss / total
        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:3d}/{epochs}  recon_loss={avg:.5f}")

        if avg < best_loss:
            best_loss = avg
            torch.save(encoder.state_dict(), save_path)

    print(f"  ✓ Stage1 最优模型已保存 → {save_path}")
