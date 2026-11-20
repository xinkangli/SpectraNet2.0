import os
import math
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# =========================================================
# Utils
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def try_parse_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def zscore_per_sample(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x))
    return (x - mu) / (sd + eps)


def linear_resample_with_mask(
    src_wavenumbers: np.ndarray,
    src_values: np.ndarray,
    target_min: float,
    target_max: float,
    target_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    x_old = np.asarray(src_wavenumbers, dtype=np.float32)
    y_old = np.asarray(src_values, dtype=np.float32)
    order = np.argsort(x_old)
    x_old = x_old[order]
    y_old = y_old[order]
    x_new = np.linspace(target_min, target_max, target_len, dtype=np.float32)
    valid = (x_new >= x_old.min()) & (x_new <= x_old.max())
    y_new = np.zeros_like(x_new, dtype=np.float32)
    if np.sum(valid) > 0:
        y_new[valid] = np.interp(x_new[valid], x_old, y_old)
    m_new = valid.astype(np.float32)
    return y_new, m_new


def finite_diff_1d(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, 1:] - x[:, :-1]
    dx = F.pad(dx, (1, 0), mode="constant", value=0.0)
    return dx


def build_patch_point_mask(mask_patch: torch.Tensor, patch_size: int) -> torch.Tensor:
    return mask_patch.unsqueeze(-1).expand(-1, -1, patch_size).reshape(mask_patch.shape[0], -1).float()


def patchify_1d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, L = x.shape
    assert L % patch_size == 0
    N = L // patch_size
    return x.view(B, N, patch_size)


def unpatchify_1d(patches: torch.Tensor) -> torch.Tensor:
    B, N, P = patches.shape
    return patches.reshape(B, N * P)


def generate_span_mask(num_patches: int, mask_ratio: float, max_span: int = 8) -> torch.Tensor:
    num_mask = int(round(num_patches * mask_ratio))
    mask = torch.zeros(num_patches, dtype=torch.bool)
    if num_mask <= 0:
        return mask
    masked = 0
    tries = 0
    while masked < num_mask and tries < num_patches * 10:
        span_len = random.randint(1, max_span)
        start = random.randint(0, num_patches - 1)
        end = min(num_patches, start + span_len)
        newly = (~mask[start:end]).sum().item()
        mask[start:end] = True
        masked += newly
        tries += 1
    if mask.sum().item() > num_mask:
        idx = torch.where(mask)[0].tolist()
        random.shuffle(idx)
        for i in idx[: mask.sum().item() - num_mask]:
            mask[i] = False
    return mask


# =========================================================
# 光谱增强 — 缓解域偏移的关键手段
# =========================================================
class SpectraAugment:
    """
    在线数据增强，模拟不同仪器/环境造成的光谱偏移。

    增强方式:
      1. baseline_shift  : 随机多项式基线 (模拟仪器漂移)
      2. amplitude_scale : 随机缩放幅度   (模拟浓度/厚度差异)
      3. additive_noise  : 高斯加性噪声
      4. random_erasing  : 随机擦除连续片段 (模拟坏点/饱和)
      5. spectral_shift  : 横向微小平移   (模拟波数校准误差)
    """
    def __init__(
        self,
        baseline_prob: float = 0.5,
        baseline_order: int = 3,
        baseline_scale: float = 0.3,
        amplitude_prob: float = 0.5,
        amplitude_range: Tuple[float, float] = (0.8, 1.2),
        noise_prob: float = 0.5,
        noise_std: float = 0.05,
        erase_prob: float = 0.3,
        erase_max_frac: float = 0.1,
        shift_prob: float = 0.3,
        shift_max_bins: int = 10,
    ):
        self.baseline_prob = baseline_prob
        self.baseline_order = baseline_order
        self.baseline_scale = baseline_scale
        self.amplitude_prob = amplitude_prob
        self.amplitude_range = amplitude_range
        self.noise_prob = noise_prob
        self.noise_std = noise_std
        self.erase_prob = erase_prob
        self.erase_max_frac = erase_max_frac
        self.shift_prob = shift_prob
        self.shift_max_bins = shift_max_bins

    @torch.no_grad()
    def __call__(self, x: torch.Tensor, m: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: [B, L] float  |  m: [B, L] float mask (1=valid)"""
        B, L = x.shape
        device = x.device

        # 1. Polynomial baseline drift
        if self.baseline_prob > 0:
            do_aug = torch.rand(B, device=device) < self.baseline_prob
            if do_aug.any():
                t = torch.linspace(-1, 1, L, device=device)
                for order in range(1, self.baseline_order + 1):
                    coef = (torch.rand(B, device=device) * 2 - 1) * self.baseline_scale
                    baseline = coef.unsqueeze(1) * (t ** order).unsqueeze(0)
                    x = x + baseline * do_aug.float().unsqueeze(1)

        # 2. Amplitude scaling
        if self.amplitude_prob > 0:
            do_aug = torch.rand(B, device=device) < self.amplitude_prob
            lo, hi = self.amplitude_range
            scale = torch.rand(B, device=device) * (hi - lo) + lo
            scale = torch.where(do_aug, scale, torch.ones_like(scale))
            x = x * scale.unsqueeze(1)

        # 3. Additive noise
        if self.noise_prob > 0:
            do_aug = torch.rand(B, device=device) < self.noise_prob
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise * do_aug.float().unsqueeze(1)

        # 4. Random erasing
        if self.erase_prob > 0:
            for b in range(B):
                if random.random() < self.erase_prob:
                    erase_len = random.randint(1, max(1, int(L * self.erase_max_frac)))
                    start = random.randint(0, L - erase_len)
                    x[b, start:start + erase_len] = 0.0
                    m[b, start:start + erase_len] = 0.0

        # 5. Spectral shift
        if self.shift_prob > 0:
            do_aug = torch.rand(B, device=device) < self.shift_prob
            for b in range(B):
                if do_aug[b]:
                    shift = random.randint(-self.shift_max_bins, self.shift_max_bins)
                    if shift != 0:
                        x[b] = torch.roll(x[b], shift, dims=0)
                        m[b] = torch.roll(m[b], shift, dims=0)

        return x, m


# =========================================================
# MMD 域对齐损失
# =========================================================
def mmd_rbf(z_src: torch.Tensor, z_tgt: torch.Tensor,
            sigma_list: List[float] = None) -> torch.Tensor:
    """
    Maximum Mean Discrepancy with RBF kernel.
    最小化源域和目标域特征分布的距离。
    z_src, z_tgt: [B, D]
    """
    if sigma_list is None:
        sigma_list = [0.1, 0.5, 1.0, 2.0, 5.0]

    def rbf(x, y, sigma):
        diff = x.unsqueeze(1) - y.unsqueeze(0)   # [Bx, By, D]
        dist = (diff ** 2).sum(-1)                # [Bx, By]
        return torch.exp(-dist / (2 * sigma ** 2))

    loss = torch.tensor(0.0, device=z_src.device)
    for sigma in sigma_list:
        k_ss = rbf(z_src, z_src, sigma).mean()
        k_tt = rbf(z_tgt, z_tgt, sigma).mean()
        k_st = rbf(z_src, z_tgt, sigma).mean()
        loss = loss + k_ss + k_tt - 2.0 * k_st
    return loss / len(sigma_list)


# =========================================================
# 梯度反转层 + 域判别器 (DANN)
# =========================================================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainDiscriminator(nn.Module):
    """判别特征属于源域(0)还是目标域(1)，通过梯度反转使编码器学习域不变特征。"""
    def __init__(self, in_dim: int = 256, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        z_rev = GradReverse.apply(z, alpha)
        return self.net(z_rev).squeeze(-1)   # [B]


# =========================================================
# Dataset
# =========================================================
class DualBranchSpectraDataset(Dataset):
    def __init__(
        self,
        xlsx_path: str,
        normalize: str = "zscore",
        shared_range: Tuple[float, float] = (1400.320416, 4000.364384),
        shared_len: int = 1600,
        full_range: Tuple[float, float] = (400.0, 4000.364384),
        full_len: int = 2048,
        dtype=np.float32,
    ):
        df = pd.read_excel(xlsx_path)

        numeric_cols = []
        for c in df.columns:
            v = try_parse_float(c)
            if v is not None:
                numeric_cols.append((c, v))
        if len(numeric_cols) == 0:
            raise ValueError(f"{xlsx_path} 中没有识别到数值型波数列")

        numeric_cols = sorted(numeric_cols, key=lambda x: x[1])
        spectral_cols = [c for c, _ in numeric_cols]
        spectral_wn = np.array([v for _, v in numeric_cols], dtype=np.float32)
        y_raw = df.iloc[:, -1].to_numpy().astype(np.int64)
        spectra = df[spectral_cols].to_numpy(dtype=dtype)

        shared_x_list, shared_m_list = [], []
        full_x_list, full_m_list = [], []
        s_min, s_max = shared_range
        f_min, f_max = full_range

        for row in spectra:
            sx, sm = linear_resample_with_mask(spectral_wn, row, s_min, s_max, shared_len)
            fx, fm = linear_resample_with_mask(spectral_wn, row, f_min, f_max, full_len)

            if normalize == "zscore":
                if sm.sum() > 0:
                    sx_norm = np.zeros_like(sx)
                    sx_norm[sm > 0] = zscore_per_sample(sx[sm > 0])
                    sx = sx_norm
                if fm.sum() > 0:
                    fx_norm = np.zeros_like(fx)
                    fx_norm[fm > 0] = zscore_per_sample(fx[fm > 0])
                    fx = fx_norm
            elif normalize == "minmax":
                if sm.sum() > 0:
                    v = sx[sm > 0]; mn, mx = v.min(), v.max()
                    tmp = np.zeros_like(sx); tmp[sm > 0] = (v - mn) / (mx - mn + 1e-8); sx = tmp
                if fm.sum() > 0:
                    v = fx[fm > 0]; mn, mx = v.min(), v.max()
                    tmp = np.zeros_like(fx); tmp[fm > 0] = (v - mn) / (mx - mn + 1e-8); fx = tmp

            shared_x_list.append(sx.astype(dtype)); shared_m_list.append(sm.astype(dtype))
            full_x_list.append(fx.astype(dtype));   full_m_list.append(fm.astype(dtype))

        uniq = np.unique(y_raw)
        self.raw_to_contig = {int(v): i for i, v in enumerate(uniq)}
        self.contig_to_raw = {i: int(v) for i, v in enumerate(uniq)}
        y_contig = np.array([self.raw_to_contig[int(v)] for v in y_raw], dtype=np.int64)

        self.shared_x = np.stack(shared_x_list, axis=0)
        self.shared_m = np.stack(shared_m_list, axis=0)
        self.full_x   = np.stack(full_x_list,   axis=0)
        self.full_m   = np.stack(full_m_list,   axis=0)
        self.y        = y_contig
        self.y_raw    = y_raw
        self.shared_len   = shared_len; self.full_len   = full_len
        self.shared_range = shared_range; self.full_range = full_range

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.shared_x[idx]),
            torch.from_numpy(self.shared_m[idx]),
            torch.from_numpy(self.full_x[idx]),
            torch.from_numpy(self.full_m[idx]),
            torch.tensor(self.y[idx],     dtype=torch.long),
            torch.tensor(self.y_raw[idx], dtype=torch.long),
        )


# =========================================================
# Model blocks
# =========================================================
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden); self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PatchTransformerEncoder(nn.Module):
    def __init__(self, input_length, patch_size, dim=256, depth=6, heads=8,
                 mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        assert input_length % patch_size == 0
        self.input_length = input_length; self.patch_size = patch_size
        self.num_patches = input_length // patch_size; self.dim = dim
        self.patch_embed_x = nn.Linear(patch_size, dim)
        self.patch_embed_m = nn.Linear(patch_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward_tokens(self, x, m):
        px = patchify_1d(x, self.patch_size); pm = patchify_1d(m, self.patch_size)
        tok = self.patch_embed_x(px) + self.patch_embed_m(pm) + self.pos
        for blk in self.blocks: tok = blk(tok)
        return self.norm(tok)

    def forward(self, x, m):
        return self.forward_tokens(x, m).mean(dim=1)


class SharedMAEDecoder(nn.Module):
    def __init__(self, num_patches, patch_size, enc_dim, dec_dim=128,
                 dec_depth=2, dec_heads=4, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.num_patches = num_patches; self.patch_size = patch_size
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim, bias=False) if enc_dim != dec_dim else nn.Identity()
        self.pos = nn.Parameter(torch.zeros(1, num_patches, dec_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(dec_dim, dec_heads, mlp_ratio, dropout) for _ in range(dec_depth)])
        self.norm = nn.LayerNorm(dec_dim); self.pred = nn.Linear(dec_dim, patch_size)

    def forward(self, visible_full_tokens, patch_mask):
        dec_tok = self.enc_to_dec(visible_full_tokens)
        mask_tok = self.mask_token.expand(dec_tok.shape[0], dec_tok.shape[1], -1)
        dec_tok = torch.where(patch_mask.unsqueeze(-1), mask_tok, dec_tok) + self.pos
        for blk in self.blocks: dec_tok = blk(dec_tok)
        return self.pred(self.norm(dec_tok))


class DualBranchModel(nn.Module):
    def __init__(
        self,
        shared_len=1600, full_len=2048,
        shared_patch=16, full_patch=16,
        shared_dim=256,  full_dim=256,
        shared_depth=6,  full_depth=6,
        heads=8, num_classes=14, fusion_dim=256, dropout=0.1,
    ):
        super().__init__()
        self.shared_encoder = PatchTransformerEncoder(
            shared_len, shared_patch, shared_dim, shared_depth, heads, dropout=0.0)
        self.full_encoder = PatchTransformerEncoder(
            full_len, full_patch, full_dim, full_depth, heads, dropout=0.0)
        self.shared_decoder = SharedMAEDecoder(
            shared_len // shared_patch, shared_patch, shared_dim, 128, 2, 4)
        self.fuse = nn.Sequential(
            nn.Linear(shared_dim + full_dim, fusion_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim), nn.GELU(),
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.shared_patch = shared_patch
        self.shared_num_patches = shared_len // shared_patch

    def encode_shared_tokens(self, sx, sm): return self.shared_encoder.forward_tokens(sx, sm)
    def encode_shared(self, sx, sm):        return self.shared_encoder(sx, sm)
    def encode_full(self, fx, fm):          return self.full_encoder(fx, fm)
    def fuse_repr(self, z_s, z_f):          return self.fuse(torch.cat([z_s, z_f], dim=1))

    def classify(self, sx, sm, fx, fm):
        z_s = self.encode_shared(sx, sm); z_f = self.encode_full(fx, fm)
        z = self.fuse_repr(z_s, z_f)
        return self.classifier(z), z_s, z_f, z

    def reconstruct_shared(self, sx, sm, mask_ratio=0.6, max_span=8):
        B = sx.shape[0]; N = self.shared_num_patches
        patch_mask = torch.stack(
            [generate_span_mask(N, mask_ratio, max_span) for _ in range(B)], dim=0
        ).to(sx.device)
        tokens = self.encode_shared_tokens(sx, sm)
        visible = tokens.clone(); visible[patch_mask] = 0.0
        return self.shared_decoder(visible, patch_mask), patch_mask


# =========================================================
# Training helpers
# =========================================================
@dataclass
class TrainConfig:
    batch_size: int = 128; lr: float = 1e-3; weight_decay: float = 1e-4
    epochs: int = 50; num_workers: int = 2; use_amp: bool = True; device: str = "cuda"


def require_device(use_gpu):
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("You requested --gpu but CUDA is not available.")
    return "cuda" if (use_gpu and torch.cuda.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")


def cosine_consistency(z1, z2):
    return 1.0 - F.cosine_similarity(z1, z2, dim=1).mean()


def save_ckpt(path, model, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"model": model.state_dict()}
    if extra: payload.update(extra)
    torch.save(payload, path)


# =========================================================
# [核心改动] 联合预训练: 源域 + 目标域无标签数据
# =========================================================
def pretrain_with_target(
    model: DualBranchModel,
    src_loader: DataLoader,
    tgt_loader: DataLoader,       # 目标域无标签 DataLoader
    cfg: TrainConfig,
    mask_ratio: float = 0.6,
    max_span: int = 8,
    lambda_der: float = 0.2,
    lambda_br: float = 0.5,
    lambda_mmd: float = 1.0,     # MMD 域对齐权重
    lambda_dann: float = 0.0,    # DANN 域对抗权重 (0=关闭)
    use_augment: bool = True,
    save_path: str = "checkpoints/dual_pretrain_da.pt",
):
    """
    改进的预训练，包含三层域适应机制:
      1. 源域 MAE + 目标域 MAE  → 编码器同时学两个域的内部结构
      2. MMD 对齐               → 拉近源/目标特征分布
      3. DANN 域对抗 (可选)     → 对抗式域混淆
    """
    model.to(cfg.device)
    domain_disc = None
    if lambda_dann > 0:
        domain_disc = DomainDiscriminator(in_dim=model.classifier.in_features).to(cfg.device)

    augment = SpectraAugment() if use_augment else None
    all_params = list(model.parameters())
    if domain_disc is not None:
        all_params += list(domain_disc.parameters())

    opt = torch.optim.AdamW(all_params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and cfg.device == "cuda"))

    tgt_iter = iter(tgt_loader)
    best = 1e18

    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        p = ep / cfg.epochs
        dann_alpha = 2.0 / (1.0 + math.exp(-10 * p)) - 1.0   # 渐进式增强

        for batch in src_loader:
            shared_x, shared_m, full_x, full_m, _, _ = batch
            shared_x = shared_x.to(cfg.device).float(); shared_m = shared_m.to(cfg.device).float()
            full_x   = full_x.to(cfg.device).float();   full_m   = full_m.to(cfg.device).float()

            try:
                tgt_batch = next(tgt_iter)
            except StopIteration:
                tgt_iter = iter(tgt_loader)
                tgt_batch = next(tgt_iter)

            t_sx, t_sm, t_fx, t_fm, _, _ = tgt_batch
            t_sx = t_sx.to(cfg.device).float(); t_sm = t_sm.to(cfg.device).float()
            t_fx = t_fx.to(cfg.device).float(); t_fm = t_fm.to(cfg.device).float()

            if augment is not None:
                shared_x, shared_m = augment(shared_x.clone(), shared_m.clone())
                full_x,   full_m   = augment(full_x.clone(),   full_m.clone())
                t_sx, t_sm         = augment(t_sx.clone(),     t_sm.clone())
                t_fx, t_fm         = augment(t_fx.clone(),     t_fm.clone())

            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and cfg.device == "cuda")):
                # --- 源域 MAE ---
                pred_p, pm = model.reconstruct_shared(shared_x, shared_m, mask_ratio, max_span)
                tgt_p = patchify_1d(shared_x, model.shared_patch)
                pmf = pm.float().unsqueeze(-1)
                loss = ((pred_p - tgt_p) ** 2 * pmf).sum() / (pmf.sum() * tgt_p.shape[-1] + 1e-8)

                if lambda_der > 0:
                    pred_shared = unpatchify_1d(pred_p)
                    pm_pt = build_patch_point_mask(pm, model.shared_patch).to(cfg.device)
                    der = ((finite_diff_1d(pred_shared) - finite_diff_1d(shared_x)).abs() * pm_pt).sum() \
                          / (pm_pt.sum() + 1e-8)
                    loss = loss + lambda_der * der

                z_s = model.encode_shared(shared_x, shared_m)
                z_f = model.encode_full(full_x, full_m)
                if lambda_br > 0:
                    loss = loss + lambda_br * cosine_consistency(z_s, z_f)

                # --- 目标域 MAE (关键: 让编码器学习目标域结构) ---
                pred_pt, pmt = model.reconstruct_shared(t_sx, t_sm, mask_ratio, max_span)
                tgt_pt = patchify_1d(t_sx, model.shared_patch)
                pmft = pmt.float().unsqueeze(-1)
                rec_tgt = ((pred_pt - tgt_pt) ** 2 * pmft).sum() / (pmft.sum() * tgt_pt.shape[-1] + 1e-8)
                loss = loss + rec_tgt

                z_ts = model.encode_shared(t_sx, t_sm)
                z_tf = model.encode_full(t_fx, t_fm)
                if lambda_br > 0:
                    loss = loss + lambda_br * cosine_consistency(z_ts, z_tf)

                # --- MMD 域对齐 ---
                if lambda_mmd > 0:
                    z_sf = model.fuse_repr(z_s, z_f)
                    z_tf2 = model.fuse_repr(z_ts, z_tf)
                    loss = loss + lambda_mmd * mmd_rbf(z_sf, z_tf2)

                # --- DANN 域对抗 ---
                if lambda_dann > 0 and domain_disc is not None:
                    z_sf = model.fuse_repr(z_s, z_f)
                    z_tf2 = model.fuse_repr(z_ts, z_tf)
                    z_all = torch.cat([z_sf, z_tf2], dim=0)
                    d_labels = torch.cat([
                        torch.zeros(z_sf.shape[0], device=cfg.device),
                        torch.ones(z_tf2.shape[0], device=cfg.device),
                    ])
                    dann_loss = F.binary_cross_entropy_with_logits(
                        domain_disc(z_all, alpha=dann_alpha), d_labels)
                    loss = loss + lambda_dann * dann_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        avg = float(np.mean(losses))
        print(f"[Pretrain+DA] epoch {ep:03d}/{cfg.epochs} loss={avg:.6f}", flush=True)
        if avg < best:
            best = avg
            save_ckpt(save_path, model)
            print(f"Saved best pretrain+DA: {save_path}", flush=True)


def pretrain_source(
    model: DualBranchModel,
    loader: DataLoader,
    cfg: TrainConfig,
    mask_ratio=0.6, max_span=8, lambda_der=0.2, lambda_br=0.5,
    use_augment=True, save_path="checkpoints/dual_pretrain.pt",
):
    model.to(cfg.device); model.train()
    augment = SpectraAugment() if use_augment else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and cfg.device == "cuda"))
    best = 1e18

    for ep in range(1, cfg.epochs + 1):
        losses = []
        for shared_x, shared_m, full_x, full_m, _, _ in loader:
            shared_x = shared_x.to(cfg.device).float(); shared_m = shared_m.to(cfg.device).float()
            full_x   = full_x.to(cfg.device).float();   full_m   = full_m.to(cfg.device).float()
            if augment:
                shared_x, shared_m = augment(shared_x.clone(), shared_m.clone())
                full_x, full_m     = augment(full_x.clone(),   full_m.clone())

            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and cfg.device == "cuda")):
                pred_p, pm = model.reconstruct_shared(shared_x, shared_m, mask_ratio, max_span)
                tgt_p = patchify_1d(shared_x, model.shared_patch)
                pmf = pm.float().unsqueeze(-1)
                loss = ((pred_p - tgt_p) ** 2 * pmf).sum() / (pmf.sum() * tgt_p.shape[-1] + 1e-8)
                if lambda_der > 0:
                    pred_shared = unpatchify_1d(pred_p)
                    pm_pt = build_patch_point_mask(pm, model.shared_patch).to(cfg.device)
                    der = ((finite_diff_1d(pred_shared) - finite_diff_1d(shared_x)).abs() * pm_pt).sum() \
                          / (pm_pt.sum() + 1e-8)
                    loss = loss + lambda_der * der
                if lambda_br > 0:
                    loss = loss + lambda_br * cosine_consistency(
                        model.encode_shared(shared_x, shared_m),
                        model.encode_full(full_x, full_m))

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        avg = float(np.mean(losses))
        print(f"[Pretrain] epoch {ep:03d}/{cfg.epochs} loss={avg:.6f}", flush=True)
        if avg < best:
            best = avg; save_ckpt(save_path, model)
            print(f"Saved best pretrain: {save_path}", flush=True)


# =========================================================
# Supervised train on source
# =========================================================
@torch.no_grad()
def eval_classifier(model, loader, device):
    model.eval(); ys, ps = [], []
    for shared_x, shared_m, full_x, full_m, y, _ in loader:
        logits, _, _, _ = model.classify(
            shared_x.to(device).float(), shared_m.to(device).float(),
            full_x.to(device).float(),   full_m.to(device).float())
        ps.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    return float(accuracy_score(ys, ps)), float(f1_score(ys, ps, average="macro"))


def train_source_supervised(
    model, train_loader, val_loader, cfg,
    freeze_encoders=False, use_augment=True,
    save_path="checkpoints/dual_source_cls.pt",
):
    model.to(cfg.device)
    augment = SpectraAugment() if use_augment else None
    if freeze_encoders:
        for p in model.shared_encoder.parameters(): p.requires_grad = False
        for p in model.full_encoder.parameters():   p.requires_grad = False

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and cfg.device == "cuda"))
    best = -1.0

    for ep in range(1, cfg.epochs + 1):
        model.train(); losses = []
        for shared_x, shared_m, full_x, full_m, y, _ in train_loader:
            shared_x = shared_x.to(cfg.device).float(); shared_m = shared_m.to(cfg.device).float()
            full_x   = full_x.to(cfg.device).float();   full_m   = full_m.to(cfg.device).float()
            y = y.to(cfg.device).long()
            if augment:
                shared_x, shared_m = augment(shared_x.clone(), shared_m.clone())
                full_x, full_m     = augment(full_x.clone(),   full_m.clone())
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and cfg.device == "cuda")):
                logits, _, _, _ = model.classify(shared_x, shared_m, full_x, full_m)
                loss = F.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        acc, f1m = eval_classifier(model, val_loader, cfg.device)
        print(f"[Source Sup] epoch {ep:03d}/{cfg.epochs} loss={np.mean(losses):.4f} val_acc={acc:.4f} macroF1={f1m:.4f}", flush=True)
        if acc > best:
            best = acc; save_ckpt(save_path, model)
            print(f"Saved best source cls: {save_path}", flush=True)
    return best


# =========================================================
# Zero-shot / Few-shot helpers
# =========================================================
def build_overlap_map_by_raw(source_ds, target_ds):
    overlap_raw = sorted(list(set(source_ds.raw_to_contig.keys()) & set(target_ds.raw_to_contig.keys())))
    t2s = {target_ds.raw_to_contig[r]: source_ds.raw_to_contig[r] for r in overlap_raw}
    return t2s, overlap_raw


class TargetOverlapDataset(Dataset):
    def __init__(self, target_ds, source_ds, overlap_raw):
        self.target_ds = target_ds
        self.indices = [i for i in range(len(target_ds)) if int(target_ds.y_raw[i]) in set(overlap_raw)]
        self.raw_to_source_contig = source_ds.raw_to_contig

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        shared_x, shared_m, full_x, full_m, _, y_raw = self.target_ds[idx]
        return shared_x, shared_m, full_x, full_m, torch.tensor(
            self.raw_to_source_contig[int(y_raw.item())], dtype=torch.long)


@torch.no_grad()
def eval_target_zeroshot(model, source_ds, target_ds, device, batch_size=128):
    _, overlap_raw = build_overlap_map_by_raw(source_ds, target_ds)
    if len(overlap_raw) == 0:
        raise RuntimeError("No overlapping raw labels between source and target.")
    ds = TargetOverlapDataset(target_ds, source_ds, overlap_raw)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    model.to(device).eval(); ys, ps = [], []
    for shared_x, shared_m, full_x, full_m, y in loader:
        logits, _, _, _ = model.classify(
            shared_x.to(device).float(), shared_m.to(device).float(),
            full_x.to(device).float(), full_m.to(device).float())
        ps.append(logits.argmax(1).cpu().numpy()); ys.append(y.numpy())
    ys = np.concatenate(ys); ps = np.concatenate(ps)
    acc = accuracy_score(ys, ps); f1m = f1_score(ys, ps, average="macro")
    print(f"[Zero-shot] samples={len(ds)} overlap_classes={len(overlap_raw)} acc={acc:.4f} macroF1={f1m:.4f}", flush=True)
    return float(acc), float(f1m)


def expand_classifier_to_target(source_model, source_ds, target_ds):
    target_num_classes = len(target_ds.raw_to_contig)
    new_model = DualBranchModel(
        shared_len=source_model.shared_encoder.input_length,
        full_len=source_model.full_encoder.input_length,
        shared_patch=source_model.shared_encoder.patch_size,
        full_patch=source_model.full_encoder.patch_size,
        shared_dim=source_model.shared_encoder.dim,
        full_dim=source_model.full_encoder.dim,
        shared_depth=len(source_model.shared_encoder.blocks),
        full_depth=len(source_model.full_encoder.blocks),
        heads=8, num_classes=target_num_classes,
        fusion_dim=source_model.classifier.in_features,
    )
    sd = source_model.state_dict(); new_sd = new_model.state_dict()
    for k in new_sd:
        if k.startswith("classifier."): continue
        if k in sd and sd[k].shape == new_sd[k].shape: new_sd[k] = sd[k]
    new_model.load_state_dict(new_sd, strict=False)

    t2s, overlap_raw = build_overlap_map_by_raw(source_ds, target_ds)
    with torch.no_grad():
        nn.init.trunc_normal_(new_model.classifier.weight, std=0.02)
        nn.init.constant_(new_model.classifier.bias, 0.0)
        for t_id, s_id in t2s.items():
            new_model.classifier.weight[t_id].copy_(source_model.classifier.weight[s_id])
            new_model.classifier.bias[t_id].copy_(source_model.classifier.bias[s_id])

    print(f"[Expand cls] target_classes={target_num_classes} overlap={len(overlap_raw)} novel={target_num_classes - len(overlap_raw)}", flush=True)
    return new_model


def make_kshot_indices(y_contig, k_per_class, seed=42):
    rng = np.random.RandomState(seed)
    classes = np.unique(y_contig)
    tr_idx, te_idx = [], []
    for c in classes:
        idx = np.where(y_contig == c)[0]; rng.shuffle(idx)
        k = min(k_per_class, len(idx))
        tr_idx.extend(idx[:k].tolist()); te_idx.extend(idx[k:].tolist())
    rng.shuffle(tr_idx); rng.shuffle(te_idx)
    return tr_idx, te_idx


# =========================================================
# 原型网络少样本评估 (无需梯度更新，更稳定)
# =========================================================
@torch.no_grad()
def prototype_eval(model, target_ds, device, kshot=5, batch_size=128, seed=42):
    """
    用每类 k 个样本的平均特征作为原型，余弦相似度做最近邻分类。
    基于支持集原型进行预测，无需梯度更新。
    """
    model.to(device).eval()
    tr_idx, te_idx = make_kshot_indices(target_ds.y, kshot, seed)

    def get_features(indices):
        subset = Subset(target_ds, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)
        feats, labels = [], []
        for sx, sm, fx, fm, y, _ in loader:
            _, _, _, z = model.classify(sx.to(device).float(), sm.to(device).float(),
                                        fx.to(device).float(), fm.to(device).float())
            feats.append(z.cpu()); labels.append(y)
        return torch.cat(feats), torch.cat(labels)

    tr_feats, tr_labels = get_features(tr_idx)
    te_feats, te_labels = get_features(te_idx)

    classes = torch.unique(tr_labels)
    prototypes = torch.stack([tr_feats[tr_labels == c].mean(0) for c in classes])
    sim = F.normalize(te_feats, dim=1) @ F.normalize(prototypes, dim=1).T
    pred = sim.argmax(1).numpy(); ys = te_labels.numpy()

    acc = accuracy_score(ys, pred); f1m = f1_score(ys, pred, average="macro")
    print(f"[Prototype k={kshot}] test_samples={len(te_idx)} acc={acc:.4f} macroF1={f1m:.4f}", flush=True)
    return float(acc), float(f1m)


def fewshot_adapt(
    model, target_ds, device, kshot=5, batch_size=64,
    freeze_backbone=True, epochs=50, lr=1e-3, wd=1e-4,
    use_amp=True, use_augment=True,
):
    tr_idx, te_idx = make_kshot_indices(target_ds.y, kshot)
    tr_loader = DataLoader(Subset(target_ds, tr_idx), batch_size=batch_size,
                           shuffle=True, drop_last=True, num_workers=2)
    te_loader = DataLoader(Subset(target_ds, te_idx), batch_size=batch_size,
                           shuffle=False, num_workers=2)

    model.to(device)
    augment = SpectraAugment(
        baseline_prob=0.5, amplitude_prob=0.5,
        noise_prob=0.5, noise_std=0.03,
        erase_prob=0.2, shift_prob=0.2,
    ) if use_augment else None

    if freeze_backbone:
        for p in model.shared_encoder.parameters(): p.requires_grad = False
        for p in model.full_encoder.parameters():   p.requires_grad = False
        for p in model.fuse.parameters():           p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    for ep in range(1, epochs + 1):
        model.train(); losses = []
        for sx, sm, fx, fm, y, _ in tr_loader:
            sx = sx.to(device).float(); sm = sm.to(device).float()
            fx = fx.to(device).float(); fm = fm.to(device).float()
            y = y.to(device).long()
            if augment:
                sx, sm = augment(sx.clone(), sm.clone())
                fx, fm = augment(fx.clone(), fm.clone())
            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                logits, _, _, _ = model.classify(sx, sm, fx, fm)
                loss = F.cross_entropy(logits, y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach().cpu().item()))
        sched.step()
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            acc, f1m = eval_classifier(model, te_loader, device)
            print(f"[Few-shot k={kshot}] epoch {ep:03d}/{epochs} loss={np.mean(losses):.4f} acc={acc:.4f} macroF1={f1m:.4f}", flush=True)

    return eval_classifier(model, te_loader, device)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=[
        "pretrain_source", "pretrain_with_target",
        "train_source", "eval_target_zeroshot",
        "eval_target_fewshot", "eval_prototype",
    ])
    parser.add_argument("--source_xlsx",    type=str,   default="data.xlsx")
    parser.add_argument("--target_xlsx",    type=str,   default="target.xlsx")
    parser.add_argument("--normalize",      type=str,   default="zscore", choices=["zscore","minmax","none"])
    parser.add_argument("--shared_min",     type=float, default=1400.320416)
    parser.add_argument("--shared_max",     type=float, default=4000.364384)
    parser.add_argument("--shared_len",     type=int,   default=1600)
    parser.add_argument("--full_min",       type=float, default=400.0)
    parser.add_argument("--full_max",       type=float, default=4000.364384)
    parser.add_argument("--full_len",       type=int,   default=2048)
    parser.add_argument("--shared_patch",   type=int,   default=16)
    parser.add_argument("--full_patch",     type=int,   default=16)
    parser.add_argument("--mask_ratio",     type=float, default=0.6)
    parser.add_argument("--max_span",       type=int,   default=8)
    parser.add_argument("--lambda_der",     type=float, default=0.2)
    parser.add_argument("--lambda_br",      type=float, default=0.5)
    parser.add_argument("--lambda_mmd",     type=float, default=1.0,  help="MMD 域对齐权重")
    parser.add_argument("--lambda_dann",    type=float, default=0.0,  help="DANN 域对抗权重, 0=关闭")
    parser.add_argument("--shared_dim",     type=int,   default=256)
    parser.add_argument("--full_dim",       type=int,   default=256)
    parser.add_argument("--shared_depth",   type=int,   default=6)
    parser.add_argument("--full_depth",     type=int,   default=6)
    parser.add_argument("--heads",          type=int,   default=8)
    parser.add_argument("--batch_size",     type=int,   default=128)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--wd",             type=float, default=1e-4)
    parser.add_argument("--pretrain_epochs",type=int,   default=50)
    parser.add_argument("--sup_epochs",     type=int,   default=40)
    parser.add_argument("--fewshot_epochs", type=int,   default=60)
    parser.add_argument("--kshot",          type=int,   default=5)
    parser.add_argument("--no_augment",     action="store_true")
    parser.add_argument("--freeze_backbone",action="store_true")
    parser.add_argument("--gpu",            action="store_true")
    parser.add_argument("--use_amp",        action="store_true")
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--pretrain_ckpt",    type=str, default="checkpoints/dual_pretrain.pt")
    parser.add_argument("--pretrain_da_ckpt", type=str, default="checkpoints/dual_pretrain_da.pt")
    parser.add_argument("--source_cls_ckpt",  type=str, default="checkpoints/dual_source_cls.pt")

    args = parser.parse_args()
    set_seed(args.seed)
    device = require_device(args.gpu)
    use_amp = args.use_amp or (device == "cuda")
    use_augment = not args.no_augment
    print(f"Device: {device} | AMP: {use_amp} | Augment: {use_augment}", flush=True)

    shared_range = (args.shared_min, args.shared_max)
    full_range   = (args.full_min,   args.full_max)

    source_ds = DualBranchSpectraDataset(
        xlsx_path=args.source_xlsx, normalize=args.normalize,
        shared_range=shared_range, shared_len=args.shared_len,
        full_range=full_range, full_len=args.full_len,
    )
    source_num_classes = len(source_ds.raw_to_contig)
    print(f"Source samples={len(source_ds)} classes={source_num_classes}", flush=True)

    model = DualBranchModel(
        shared_len=args.shared_len, full_len=args.full_len,
        shared_patch=args.shared_patch, full_patch=args.full_patch,
        shared_dim=args.shared_dim, full_dim=args.full_dim,
        shared_depth=args.shared_depth, full_depth=args.full_depth,
        heads=args.heads, num_classes=source_num_classes, fusion_dim=256,
    )

    idx = np.arange(len(source_ds))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=source_ds.y)
    src_train_loader = DataLoader(Subset(source_ds, tr_idx.tolist()), batch_size=args.batch_size,
                                  shuffle=True, drop_last=True, num_workers=2)
    src_val_loader   = DataLoader(Subset(source_ds, va_idx.tolist()), batch_size=args.batch_size,
                                  shuffle=False, num_workers=2)

    if args.mode == "pretrain_source":
        cfg = TrainConfig(batch_size=args.batch_size, lr=args.lr, weight_decay=args.wd,
                          epochs=args.pretrain_epochs, use_amp=use_amp, device=device)
        pretrain_source(model, src_train_loader, cfg, args.mask_ratio, args.max_span,
                        args.lambda_der, args.lambda_br, use_augment, args.pretrain_ckpt)
        return

    if args.mode == "pretrain_with_target":
        target_ds = DualBranchSpectraDataset(
            xlsx_path=args.target_xlsx, normalize=args.normalize,
            shared_range=shared_range, shared_len=args.shared_len,
            full_range=full_range, full_len=args.full_len,
        )
        print(f"Target samples={len(target_ds)} classes={len(target_ds.raw_to_contig)}", flush=True)
        tgt_loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=True,
                                drop_last=True, num_workers=2)
        cfg = TrainConfig(batch_size=args.batch_size, lr=args.lr, weight_decay=args.wd,
                          epochs=args.pretrain_epochs, use_amp=use_amp, device=device)
        pretrain_with_target(model, src_train_loader, tgt_loader, cfg,
                             args.mask_ratio, args.max_span, args.lambda_der, args.lambda_br,
                             args.lambda_mmd, args.lambda_dann, use_augment, args.pretrain_da_ckpt)
        return

    # 加载预训练权重 (优先 DA 版本)
    ckpt_to_load = args.pretrain_da_ckpt if os.path.exists(args.pretrain_da_ckpt) else args.pretrain_ckpt
    if not os.path.exists(ckpt_to_load):
        raise FileNotFoundError(f"Pretrain ckpt not found: {ckpt_to_load}")
    model.load_state_dict(torch.load(ckpt_to_load, map_location="cpu")["model"], strict=False)
    print(f"Loaded pretrain ckpt: {ckpt_to_load}", flush=True)

    if args.mode == "train_source":
        cfg = TrainConfig(batch_size=args.batch_size, lr=args.lr, weight_decay=args.wd,
                          epochs=args.sup_epochs, use_amp=use_amp, device=device)
        best = train_source_supervised(model, src_train_loader, src_val_loader, cfg,
                                       args.freeze_backbone, use_augment, args.source_cls_ckpt)
        print(f"Best source val acc: {best:.4f}", flush=True)
        return

    if not os.path.exists(args.source_cls_ckpt):
        raise FileNotFoundError(f"Source classifier ckpt not found: {args.source_cls_ckpt}")
    model.load_state_dict(torch.load(args.source_cls_ckpt, map_location="cpu")["model"], strict=False)
    print(f"Loaded source cls ckpt: {args.source_cls_ckpt}", flush=True)

    target_ds = DualBranchSpectraDataset(
        xlsx_path=args.target_xlsx, normalize=args.normalize,
        shared_range=shared_range, shared_len=args.shared_len,
        full_range=full_range, full_len=args.full_len,
    )
    print(f"Target samples={len(target_ds)} classes={len(target_ds.raw_to_contig)}", flush=True)

    if args.mode == "eval_target_zeroshot":
        eval_target_zeroshot(model, source_ds, target_ds, device, args.batch_size)
        return

    if args.mode == "eval_prototype":
        prototype_eval(model, target_ds, device, args.kshot, args.batch_size)
        return

    if args.mode == "eval_target_fewshot":
        target_model = expand_classifier_to_target(model, source_ds, target_ds)

        print("\n[Stage 1] Prototype eval (zero gradient, baseline)", flush=True)
        prototype_eval(target_model, target_ds, device, args.kshot)

        print("\n[Stage 2] Freeze backbone, train classifier only", flush=True)
        acc1, f11 = fewshot_adapt(target_model, target_ds, device, args.kshot,
                                  min(args.batch_size, 64), True, args.fewshot_epochs,
                                  1e-3, args.wd, use_amp, use_augment)

        print("\n[Stage 3] Unfreeze full model, micro fine-tune", flush=True)
        acc2, f12 = fewshot_adapt(target_model, target_ds, device, args.kshot,
                                  min(args.batch_size, 64), False,
                                  max(20, args.fewshot_epochs // 2),
                                  1e-4, args.wd, use_amp, use_augment)

        print(f"[Few-shot Final] stage2_acc={acc1:.4f} f1={f11:.4f} | stage3_acc={acc2:.4f} f1={f12:.4f}", flush=True)


if __name__ == "__main__":
    main()
