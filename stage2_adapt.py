"""
stage2_adapt.py
───────────────
Stage 2：新批次自监督预适应
  mode="autoencoder"  → 重建适应
  mode="simclr"       → 对比学习
  mode="masked"       → Masked Reconstruction
  mode="byol"         → BYOL 风格自监督
  mode="all"          → 依次运行全部四种（推荐）
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import ProjectionHead, PredictionHead, DecoderHead
from data_utils import UnlabeledDataset


# ──────────────────────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────────────────────

def run_stage2(encoder, X_new, signal_len,
               mode="all",
               epochs=30, lr=5e-4, batch_size=32,
               device="cpu",
               save_path="checkpoints/stage2_encoder.pt"):

    print(f"  [Stage2] mode={mode}  new_batch_size={len(X_new)}  epochs={epochs}")
    encoder = encoder.to(device)

    modes = ["autoencoder", "simclr", "masked", "byol"] if mode == "all" else [mode]

    for m in modes:
        print(f"\n  ── Stage2-{m} ──")
        if m == "autoencoder":
            _adapt_autoencoder(encoder, X_new, signal_len, epochs, lr, batch_size, device)
        elif m == "simclr":
            _adapt_simclr(encoder, X_new, epochs, lr, batch_size, device)
        elif m == "masked":
            _adapt_masked(encoder, X_new, signal_len, epochs, lr, batch_size, device)
        elif m == "byol":
            _adapt_byol(encoder, X_new, epochs, lr, batch_size, device)
        else:
            raise ValueError(f"未知 stage2 mode: {m}")

    torch.save(encoder.state_dict(), save_path)
    print(f"\n  ✓ Stage2 适应后 encoder 已保存 → {save_path}")


# ──────────────────────────────────────────────────────────────
#  数据增强（1D 光谱）
# ──────────────────────────────────────────────────────────────

def _augment(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, 1, L)
    随机组合：高斯噪声 + 随机缩放 + 随机 shift
    """
    B, C, L = x.shape
    # 高斯噪声
    x = x + 0.02 * torch.randn_like(x)
    # 随机幅值缩放 [0.8, 1.2]
    scale = 0.8 + 0.4 * torch.rand(B, 1, 1, device=x.device)
    x = x * scale
    # 随机基线偏移 [-0.1, 0.1]
    shift = (torch.rand(B, 1, 1, device=x.device) - 0.5) * 0.2
    x = x + shift
    # 随机时间 roll（模拟波形位移）
    roll = torch.randint(-L // 10, L // 10 + 1, (1,)).item()
    x = torch.roll(x, roll, dims=-1)
    return x.clamp(-1.5, 1.5)


# ──────────────────────────────────────────────────────────────
#  (1) AutoEncoder 适应
# ──────────────────────────────────────────────────────────────

def _adapt_autoencoder(encoder, X, signal_len, epochs, lr, batch_size, device):
    loader  = _loader(X, batch_size)
    decoder = DecoderHead(encoder.out_dim, signal_len).to(device)
    optim   = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    for ep in range(1, epochs + 1):
        encoder.train(); decoder.train()
        losses = []
        for (xb,) in loader:
            xb    = xb.to(device)
            xb_aug = _augment(xb)              # 加噪输入，重建干净信号
            z     = encoder(xb_aug)
            x_hat = decoder(z)
            loss  = F.mse_loss(x_hat, xb)
            optim.zero_grad(); loss.backward(); optim.step()
            losses.append(loss.item())
        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:3d}  recon={sum(losses)/len(losses):.5f}")


# ──────────────────────────────────────────────────────────────
#  (2) SimCLR 对比学习
# ──────────────────────────────────────────────────────────────

def _adapt_simclr(encoder, X, epochs, lr, batch_size, device, temperature=0.5):
    loader = _loader(X, batch_size)
    proj   = ProjectionHead(encoder.out_dim, proj_dim=64).to(device)
    optim  = torch.optim.Adam(
        list(encoder.parameters()) + list(proj.parameters()), lr=lr
    )
    for ep in range(1, epochs + 1):
        encoder.train(); proj.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            # 两路增强
            z1 = proj(encoder(_augment(xb)))   # (B, proj_dim)  归一化
            z2 = proj(encoder(_augment(xb)))

            loss = _nt_xent(z1, z2, temperature)
            optim.zero_grad(); loss.backward(); optim.step()
            losses.append(loss.item())
        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:3d}  NT-Xent={sum(losses)/len(losses):.4f}")


def _nt_xent(z1, z2, temperature=0.5):
    """NT-Xent 对比损失（SimCLR）"""
    B   = z1.size(0)
    z   = torch.cat([z1, z2], dim=0)           # (2B, d)
    sim = torch.mm(z, z.T) / temperature        # (2B, 2B)
    # 去掉对角线（自身相似度）
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -1e9)

    # 正样本下标：i→i+B，i+B→i
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B,     device=z.device),
    ])
    return F.cross_entropy(sim, labels)


# ──────────────────────────────────────────────────────────────
#  (3) Masked Reconstruction
# ──────────────────────────────────────────────────────────────

def _adapt_masked(encoder, X, signal_len, epochs, lr, batch_size, device,
                  mask_ratio=0.3):
    loader  = _loader(X, batch_size)
    decoder = DecoderHead(encoder.out_dim, signal_len).to(device)
    optim   = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    for ep in range(1, epochs + 1):
        encoder.train(); decoder.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)                 # (B,1,L)

            # 随机掩码
            xb_masked, mask = _random_mask(xb, mask_ratio)  # mask: (B,1,L)

            z     = encoder(xb_masked)
            x_hat = decoder(z)

            # 仅对被掩盖位置计算损失
            loss = (F.mse_loss(x_hat, xb, reduction='none') * mask).mean()
            optim.zero_grad(); loss.backward(); optim.step()
            losses.append(loss.item())
        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:3d}  masked_recon={sum(losses)/len(losses):.5f}")


def _random_mask(x, ratio):
    """
    对每条序列随机遮蔽 ratio 比例的连续区间。
    返回 masked_x (被遮蔽位置置0) 和 mask (1=被遮蔽)。
    """
    B, C, L    = x.shape
    mask_len   = int(L * ratio)
    mask       = torch.zeros(B, 1, L, device=x.device)
    start      = torch.randint(0, L - mask_len, (B,))
    for i, s in enumerate(start):
        mask[i, :, s:s + mask_len] = 1.0
    return x * (1 - mask), mask


# ──────────────────────────────────────────────────────────────
#  (4) BYOL 风格自监督
# ──────────────────────────────────────────────────────────────

def _adapt_byol(encoder, X, epochs, lr, batch_size, device, ema_tau=0.996):
    """
    BYOL：online encoder 预测 target encoder 的表示，target 用 EMA 更新。
    无需负样本，批量可以较小。
    """
    loader = _loader(X, batch_size)

    # Online 分支：encoder + 投影头 + 预测头
    online_proj = ProjectionHead(encoder.out_dim, proj_dim=64).to(device)
    pred_head   = PredictionHead(64, hidden=32).to(device)

    # Target 分支（EMA 副本，不参与反向传播）
    target_enc  = copy.deepcopy(encoder).to(device)
    target_proj = copy.deepcopy(online_proj).to(device)
    _set_no_grad(target_enc)
    _set_no_grad(target_proj)

    optim = torch.optim.Adam(
        list(encoder.parameters()) +
        list(online_proj.parameters()) +
        list(pred_head.parameters()),
        lr=lr,
    )

    for ep in range(1, epochs + 1):
        encoder.train(); online_proj.train(); pred_head.train()
        losses = []
        for (xb,) in loader:
            xb = xb.to(device)
            v1, v2 = _augment(xb), _augment(xb)

            # Online
            z1_online = pred_head(online_proj(encoder(v1)))
            z2_online = pred_head(online_proj(encoder(v2)))

            # Target（no grad）
            with torch.no_grad():
                z1_target = target_proj(target_enc(v1))
                z2_target = target_proj(target_enc(v2))

            loss = (_byol_loss(z1_online, z2_target) +
                    _byol_loss(z2_online, z1_target)) / 2

            optim.zero_grad(); loss.backward(); optim.step()

            # EMA 更新 target
            _ema_update(encoder,     target_enc,  ema_tau)
            _ema_update(online_proj, target_proj, ema_tau)

            losses.append(loss.item())

        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:3d}  BYOL={sum(losses)/len(losses):.5f}")


def _byol_loss(p, z):
    """负余弦相似度"""
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return -(p * z).sum(dim=-1).mean()


def _ema_update(online, target, tau):
    for po, pt in zip(online.parameters(), target.parameters()):
        pt.data = tau * pt.data + (1 - tau) * po.data


def _set_no_grad(model):
    for p in model.parameters():
        p.requires_grad = False


# ──────────────────────────────────────────────────────────────
#  工具
# ──────────────────────────────────────────────────────────────

def _loader(X, batch_size):
    from data_utils import UnlabeledDataset
    ds = UnlabeledDataset(X)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
