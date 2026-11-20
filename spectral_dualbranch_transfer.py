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
    """
    对单条光谱:
      1) 目标网格统一到 [target_min, target_max]
      2) 在原始覆盖范围内做线性插值
      3) 不在原始覆盖范围内的位置置0，并mask=0
    返回:
      resampled_values: [target_len]
      valid_mask:       [target_len], 1=有真实观测, 0=缺失
    """
    x_old = np.asarray(src_wavenumbers, dtype=np.float32)
    y_old = np.asarray(src_values, dtype=np.float32)

    # 保证单调递增
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
    """
    x: [B, L]
    """
    dx = x[:, 1:] - x[:, :-1]
    dx = F.pad(dx, (1, 0), mode="constant", value=0.0)
    return dx


def build_patch_point_mask(mask_patch: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    mask_patch: [B, N] bool
    return point mask [B, N*P]
    """
    return mask_patch.unsqueeze(-1).expand(-1, -1, patch_size).reshape(mask_patch.shape[0], -1).float()


def patchify_1d(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    B, L = x.shape
    assert L % patch_size == 0, f"L={L} not divisible by patch_size={patch_size}"
    N = L // patch_size
    return x.view(B, N, patch_size)


def unpatchify_1d(patches: torch.Tensor) -> torch.Tensor:
    B, N, P = patches.shape
    return patches.reshape(B, N * P)


def generate_span_mask(num_patches: int, mask_ratio: float, max_span: int = 8) -> torch.Tensor:
    """
    返回 [N] bool, True=masked
    """
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

    # overshoot correction
    if mask.sum().item() > num_mask:
        idx = torch.where(mask)[0].tolist()
        random.shuffle(idx)
        for i in idx[: mask.sum().item() - num_mask]:
            mask[i] = False

    return mask


# =========================================================
# Dataset
# =========================================================
class DualBranchSpectraDataset(Dataset):
    """
    输入 Excel:
      - 数值列名: 波数
      - 最后一列: y
      - 其他非数值列: 自动忽略

    输出:
      shared_x:     [1600]   1400-4000
      shared_mask:  [1600]   这里通常接近全1，但保留兼容性
      full_x:       [2048]   400-4000
      full_mask:    [2048]
      y_contig
      y_raw
    """
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

        # 默认最后一列是 y
        y_raw = df.iloc[:, -1].to_numpy().astype(np.int64)

        # 光谱矩阵只取数值型波数列
        spectra = df[spectral_cols].to_numpy(dtype=dtype)

        shared_x_list = []
        shared_m_list = []
        full_x_list = []
        full_m_list = []

        s_min, s_max = shared_range
        f_min, f_max = full_range

        for row in spectra:
            sx, sm = linear_resample_with_mask(spectral_wn, row, s_min, s_max, shared_len)
            fx, fm = linear_resample_with_mask(spectral_wn, row, f_min, f_max, full_len)

            if normalize == "zscore":
                # 仅对有效区做zscore，缺失区保持0
                if sm.sum() > 0:
                    sx_valid = sx[sm > 0]
                    sx_norm = np.zeros_like(sx)
                    sx_norm[sm > 0] = zscore_per_sample(sx_valid)
                    sx = sx_norm
                if fm.sum() > 0:
                    fx_valid = fx[fm > 0]
                    fx_norm = np.zeros_like(fx)
                    fx_norm[fm > 0] = zscore_per_sample(fx_valid)
                    fx = fx_norm
            elif normalize == "minmax":
                if sm.sum() > 0:
                    v = sx[sm > 0]
                    mn, mx = v.min(), v.max()
                    tmp = np.zeros_like(sx)
                    tmp[sm > 0] = (v - mn) / (mx - mn + 1e-8)
                    sx = tmp
                if fm.sum() > 0:
                    v = fx[fm > 0]
                    mn, mx = v.min(), v.max()
                    tmp = np.zeros_like(fx)
                    tmp[fm > 0] = (v - mn) / (mx - mn + 1e-8)
                    fx = tmp
            elif normalize == "none":
                pass
            else:
                raise ValueError("normalize must be one of: zscore, minmax, none")

            shared_x_list.append(sx.astype(dtype))
            shared_m_list.append(sm.astype(dtype))
            full_x_list.append(fx.astype(dtype))
            full_m_list.append(fm.astype(dtype))

        uniq = np.unique(y_raw)
        self.raw_to_contig = {int(v): i for i, v in enumerate(uniq)}
        self.contig_to_raw = {i: int(v) for i, v in enumerate(uniq)}
        y_contig = np.array([self.raw_to_contig[int(v)] for v in y_raw], dtype=np.int64)

        self.shared_x = np.stack(shared_x_list, axis=0)
        self.shared_m = np.stack(shared_m_list, axis=0)
        self.full_x = np.stack(full_x_list, axis=0)
        self.full_m = np.stack(full_m_list, axis=0)
        self.y = y_contig
        self.y_raw = y_raw

        self.shared_len = shared_len
        self.full_len = full_len
        self.shared_range = shared_range
        self.full_range = full_range

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.shared_x[idx]),
            torch.from_numpy(self.shared_m[idx]),
            torch.from_numpy(self.full_x[idx]),
            torch.from_numpy(self.full_m[idx]),
            torch.tensor(self.y[idx], dtype=torch.long),
            torch.tensor(self.y_raw[idx], dtype=torch.long),
        )


# =========================================================
# Model blocks
# =========================================================
class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PatchTransformerEncoder(nn.Module):
    """
    支持:
      - 输入 x: [B, L]
      - 输入 mask/value mask: [B, L]
    mask 会作为一个额外通道嵌入到 patch token 中
    """
    def __init__(
        self,
        input_length: int,
        patch_size: int,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert input_length % patch_size == 0
        self.input_length = input_length
        self.patch_size = patch_size
        self.num_patches = input_length // patch_size
        self.dim = dim

        self.patch_embed_x = nn.Linear(patch_size, dim)
        self.patch_embed_m = nn.Linear(patch_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward_tokens(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        px = patchify_1d(x, self.patch_size)   # [B,N,P]
        pm = patchify_1d(m, self.patch_size)   # [B,N,P]
        tok = self.patch_embed_x(px) + self.patch_embed_m(pm) + self.pos
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        return tok

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        tok = self.forward_tokens(x, m)
        return tok.mean(dim=1)   # global representation


class SharedMAEDecoder(nn.Module):
    """
    仅对 shared branch 做掩码重建
    """
    def __init__(
        self,
        num_patches: int,
        patch_size: int,
        enc_dim: int,
        dec_dim: int = 128,
        dec_depth: int = 2,
        dec_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.enc_to_dec = nn.Linear(enc_dim, dec_dim, bias=False) if enc_dim != dec_dim else nn.Identity()
        self.pos = nn.Parameter(torch.zeros(1, num_patches, dec_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlock(dec_dim, dec_heads, mlp_ratio, dropout) for _ in range(dec_depth)
        ])
        self.norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, patch_size)

    def forward(self, visible_full_tokens: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        """
        visible_full_tokens: [B, N, enc_dim]
        patch_mask:          [B, N] True=masked
        """
        dec_tok = self.enc_to_dec(visible_full_tokens)
        mask_tok = self.mask_token.expand(dec_tok.shape[0], dec_tok.shape[1], -1)
        dec_tok = torch.where(patch_mask.unsqueeze(-1), mask_tok, dec_tok)
        dec_tok = dec_tok + self.pos
        for blk in self.blocks:
            dec_tok = blk(dec_tok)
        dec_tok = self.norm(dec_tok)
        pred_patches = self.pred(dec_tok)   # [B,N,P]
        return pred_patches


class DualBranchModel(nn.Module):
    def __init__(
        self,
        shared_len: int = 1600,
        full_len: int = 2048,
        shared_patch: int = 16,
        full_patch: int = 16,
        shared_dim: int = 256,
        full_dim: int = 256,
        shared_depth: int = 6,
        full_depth: int = 6,
        heads: int = 8,
        num_classes: int = 14,
        fusion_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.shared_encoder = PatchTransformerEncoder(
            input_length=shared_len,
            patch_size=shared_patch,
            dim=shared_dim,
            depth=shared_depth,
            heads=heads,
            dropout=0.0,
        )
        self.full_encoder = PatchTransformerEncoder(
            input_length=full_len,
            patch_size=full_patch,
            dim=full_dim,
            depth=full_depth,
            heads=heads,
            dropout=0.0,
        )

        self.shared_decoder = SharedMAEDecoder(
            num_patches=shared_len // shared_patch,
            patch_size=shared_patch,
            enc_dim=shared_dim,
            dec_dim=128,
            dec_depth=2,
            dec_heads=4,
        )

        self.fuse = nn.Sequential(
            nn.Linear(shared_dim + full_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)

        self.shared_patch = shared_patch
        self.shared_num_patches = shared_len // shared_patch

    def encode_shared_tokens(self, shared_x: torch.Tensor, shared_m: torch.Tensor) -> torch.Tensor:
        return self.shared_encoder.forward_tokens(shared_x, shared_m)

    def encode_shared(self, shared_x: torch.Tensor, shared_m: torch.Tensor) -> torch.Tensor:
        return self.shared_encoder(shared_x, shared_m)

    def encode_full(self, full_x: torch.Tensor, full_m: torch.Tensor) -> torch.Tensor:
        return self.full_encoder(full_x, full_m)

    def fuse_repr(self, z_s: torch.Tensor, z_f: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([z_s, z_f], dim=1))

    def classify(self, shared_x: torch.Tensor, shared_m: torch.Tensor, full_x: torch.Tensor, full_m: torch.Tensor):
        z_s = self.encode_shared(shared_x, shared_m)
        z_f = self.encode_full(full_x, full_m)
        z = self.fuse_repr(z_s, z_f)
        logits = self.classifier(z)
        return logits, z_s, z_f, z

    def reconstruct_shared(self, shared_x: torch.Tensor, shared_m: torch.Tensor, mask_ratio: float = 0.6, max_span: int = 8):
        """
        shared branch 做 MAE
        返回:
          pred_patches, patch_mask
        """
        B, L = shared_x.shape
        N = self.shared_num_patches

        patch_mask = []
        for _ in range(B):
            patch_mask.append(generate_span_mask(N, mask_ratio=mask_ratio, max_span=max_span))
        patch_mask = torch.stack(patch_mask, dim=0).to(shared_x.device)  # [B,N]

        # full tokens
        tokens = self.encode_shared_tokens(shared_x, shared_m)   # [B,N,D]

        # masked patches in encoder output set to zero for decoder input
        visible_full = tokens.clone()
        visible_full[patch_mask] = 0.0

        pred_patches = self.shared_decoder(visible_full, patch_mask)
        return pred_patches, patch_mask


# =========================================================
# Training helpers
# =========================================================
@dataclass
class TrainConfig:
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    num_workers: int = 2
    use_amp: bool = True
    device: str = "cuda"


def require_device(use_gpu: bool) -> str:
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError("You requested --gpu but CUDA is not available.")
    return "cuda" if (use_gpu and torch.cuda.is_available()) else ("cuda" if torch.cuda.is_available() else "cpu")


def cosine_consistency(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    return 1.0 - F.cosine_similarity(z1, z2, dim=1).mean()


def save_ckpt(path: str, model: nn.Module):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model": model.state_dict()}, path)


# =========================================================
# Pretrain
# =========================================================
def pretrain_source(
    model: DualBranchModel,
    loader: DataLoader,
    cfg: TrainConfig,
    mask_ratio: float = 0.6,
    max_span: int = 8,
    lambda_der: float = 0.2,
    lambda_br: float = 0.5,
    save_path: str = "checkpoints/dual_pretrain.pt",
):
    model.to(cfg.device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and cfg.device == "cuda"))

    best = 1e18
    for ep in range(1, cfg.epochs + 1):
        losses = []

        for shared_x, shared_m, full_x, full_m, _, _ in loader:
            shared_x = shared_x.to(cfg.device).float()
            shared_m = shared_m.to(cfg.device).float()
            full_x = full_x.to(cfg.device).float()
            full_m = full_m.to(cfg.device).float()

            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and cfg.device == "cuda")):
                # shared MAE reconstruction
                pred_patches, patch_mask = model.reconstruct_shared(
                    shared_x, shared_m, mask_ratio=mask_ratio, max_span=max_span
                )
                tgt_patches = patchify_1d(shared_x, model.shared_patch)
                patch_mask_f = patch_mask.float().unsqueeze(-1)

                rec = (pred_patches - tgt_patches) ** 2
                rec = (rec * patch_mask_f).sum() / (patch_mask_f.sum() * tgt_patches.shape[-1] + 1e-8)

                loss = rec

                # derivative consistency on masked points
                if lambda_der > 0:
                    pred_shared = unpatchify_1d(pred_patches)
                    dx_t = finite_diff_1d(shared_x)
                    dx_p = finite_diff_1d(pred_shared)
                    point_mask = build_patch_point_mask(patch_mask, model.shared_patch).to(cfg.device)
                    der = (dx_p - dx_t).abs()
                    der = (der * point_mask).sum() / (point_mask.sum() + 1e-8)
                    loss = loss + lambda_der * der

                # branch consistency
                if lambda_br > 0:
                    z_s = model.encode_shared(shared_x, shared_m)
                    z_f = model.encode_full(full_x, full_m)
                    br = cosine_consistency(z_s, z_f)
                    loss = loss + lambda_br * br

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        avg_loss = float(np.mean(losses))
        print(f"[Pretrain] epoch {ep:03d}/{cfg.epochs} loss={avg_loss:.6f}", flush=True)

        if avg_loss < best:
            best = avg_loss
            save_ckpt(save_path, model)
            print(f"Saved best pretrain: {save_path}", flush=True)


# =========================================================
# Supervised train on source
# =========================================================
@torch.no_grad()
def eval_classifier(model: DualBranchModel, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    ys, ps = [], []

    for shared_x, shared_m, full_x, full_m, y, _ in loader:
        shared_x = shared_x.to(device).float()
        shared_m = shared_m.to(device).float()
        full_x = full_x.to(device).float()
        full_m = full_m.to(device).float()

        logits, _, _, _ = model.classify(shared_x, shared_m, full_x, full_m)
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro")
    return float(acc), float(f1m)


def train_source_supervised(
    model: DualBranchModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    freeze_encoders: bool = False,
    save_path: str = "checkpoints/dual_source_cls.pt",
):
    model.to(cfg.device)

    if freeze_encoders:
        for p in model.shared_encoder.parameters():
            p.requires_grad = False
        for p in model.full_encoder.parameters():
            p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and cfg.device == "cuda"))

    best = -1.0
    for ep in range(1, cfg.epochs + 1):
        model.train()
        losses = []

        for shared_x, shared_m, full_x, full_m, y, _ in train_loader:
            shared_x = shared_x.to(cfg.device).float()
            shared_m = shared_m.to(cfg.device).float()
            full_x = full_x.to(cfg.device).float()
            full_m = full_m.to(cfg.device).float()
            y = y.to(cfg.device).long()

            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and cfg.device == "cuda")):
                logits, _, _, _ = model.classify(shared_x, shared_m, full_x, full_m)
                loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        acc, f1m = eval_classifier(model, val_loader, cfg.device)
        print(f"[Source Sup] epoch {ep:03d}/{cfg.epochs} loss={np.mean(losses):.4f} val_acc={acc:.4f} macroF1={f1m:.4f}", flush=True)

        if acc > best:
            best = acc
            save_ckpt(save_path, model)
            print(f"Saved best source cls: {save_path}", flush=True)

    return best


# =========================================================
# Zero-shot / Few-shot helpers
# =========================================================
def build_overlap_map_by_raw(source_ds: DualBranchSpectraDataset, target_ds: DualBranchSpectraDataset):
    source_raw_set = set(source_ds.raw_to_contig.keys())
    target_raw_set = set(target_ds.raw_to_contig.keys())
    overlap_raw = sorted(list(source_raw_set.intersection(target_raw_set)))

    target_contig_to_source_contig = {}
    for raw in overlap_raw:
        s_id = source_ds.raw_to_contig[int(raw)]
        t_id = target_ds.raw_to_contig[int(raw)]
        target_contig_to_source_contig[t_id] = s_id

    return target_contig_to_source_contig, overlap_raw


class TargetOverlapDataset(Dataset):
    """
    用于严格 zero-shot:
    只保留 overlap 类，并把 target label 转成 source contig label
    """
    def __init__(self, target_ds: DualBranchSpectraDataset, source_ds: DualBranchSpectraDataset, overlap_raw: List[int]):
        self.target_ds = target_ds
        self.indices = [i for i in range(len(target_ds)) if int(target_ds.y_raw[i]) in set(overlap_raw)]
        self.raw_to_source_contig = source_ds.raw_to_contig

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        shared_x, shared_m, full_x, full_m, _, y_raw = self.target_ds[idx]
        y_src = self.raw_to_source_contig[int(y_raw.item())]
        return shared_x, shared_m, full_x, full_m, torch.tensor(y_src, dtype=torch.long)


@torch.no_grad()
def eval_target_zeroshot(model: DualBranchModel, source_ds: DualBranchSpectraDataset, target_ds: DualBranchSpectraDataset, device: str, batch_size: int = 128):
    _, overlap_raw = build_overlap_map_by_raw(source_ds, target_ds)
    if len(overlap_raw) == 0:
        raise RuntimeError("No overlapping raw labels between source and target.")

    ds = TargetOverlapDataset(target_ds, source_ds, overlap_raw)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model.to(device).eval()
    ys, ps = [], []

    for shared_x, shared_m, full_x, full_m, y in loader:
        shared_x = shared_x.to(device).float()
        shared_m = shared_m.to(device).float()
        full_x = full_x.to(device).float()
        full_m = full_m.to(device).float()

        logits, _, _, _ = model.classify(shared_x, shared_m, full_x, full_m)
        pred = logits.argmax(dim=1).cpu().numpy()

        ys.append(y.numpy())
        ps.append(pred)

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)
    acc = accuracy_score(ys, ps)
    f1m = f1_score(ys, ps, average="macro")
    print(f"[Zero-shot overlap] samples={len(ds)} overlap_classes={len(overlap_raw)} acc={acc:.4f} macroF1={f1m:.4f}", flush=True)
    return float(acc), float(f1m)


def expand_classifier_to_target(
    source_model: DualBranchModel,
    source_ds: DualBranchSpectraDataset,
    target_ds: DualBranchSpectraDataset,
) -> DualBranchModel:
    """
    扩展到 target 类别空间:
      overlap 类复制 source classifier 权重
      novel 类随机初始化
    """
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
        heads=8,
        num_classes=target_num_classes,
        fusion_dim=source_model.classifier.in_features,
    )

    # 先复制 backbone / fuse
    sd = source_model.state_dict()
    new_sd = new_model.state_dict()

    for k in new_sd.keys():
        if k.startswith("classifier."):
            continue
        if k in sd and sd[k].shape == new_sd[k].shape:
            new_sd[k] = sd[k]

    new_model.load_state_dict(new_sd, strict=False)

    # 分类头复制 overlap 类
    t2s, overlap_raw = build_overlap_map_by_raw(source_ds, target_ds)
    with torch.no_grad():
        nn.init.trunc_normal_(new_model.classifier.weight, std=0.02)
        nn.init.constant_(new_model.classifier.bias, 0.0)
        for t_id, s_id in t2s.items():
            new_model.classifier.weight[t_id].copy_(source_model.classifier.weight[s_id])
            new_model.classifier.bias[t_id].copy_(source_model.classifier.bias[s_id])

    print(f"[Expand cls] target_classes={target_num_classes} overlap={len(overlap_raw)} novel={target_num_classes - len(overlap_raw)}", flush=True)
    return new_model


def make_kshot_indices(y_contig: np.ndarray, k_per_class: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    classes = np.unique(y_contig)
    tr_idx, te_idx = [], []

    for c in classes:
        idx = np.where(y_contig == c)[0]
        rng.shuffle(idx)
        k = min(k_per_class, len(idx))
        tr_idx.extend(idx[:k].tolist())
        te_idx.extend(idx[k:].tolist())

    rng.shuffle(tr_idx)
    rng.shuffle(te_idx)
    return tr_idx, te_idx


def fewshot_adapt(
    model: DualBranchModel,
    target_ds: DualBranchSpectraDataset,
    device: str,
    kshot: int = 5,
    batch_size: int = 64,
    freeze_backbone: bool = True,
    epochs: int = 50,
    lr: float = 1e-3,
    wd: float = 1e-4,
    use_amp: bool = True,
):
    tr_idx, te_idx = make_kshot_indices(target_ds.y, kshot)
    tr_ds = Subset(target_ds, tr_idx)
    te_ds = Subset(target_ds, te_idx)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)
    te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model.to(device)

    if freeze_backbone:
        for p in model.shared_encoder.parameters():
            p.requires_grad = False
        for p in model.full_encoder.parameters():
            p.requires_grad = False
        for p in model.fuse.parameters():
            p.requires_grad = False

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    for ep in range(1, epochs + 1):
        model.train()
        losses = []

        for shared_x, shared_m, full_x, full_m, y, _ in tr_loader:
            shared_x = shared_x.to(device).float()
            shared_m = shared_m.to(device).float()
            full_x = full_x.to(device).float()
            full_m = full_m.to(device).float()
            y = y.to(device).long()

            with torch.cuda.amp.autocast(enabled=(use_amp and device == "cuda")):
                logits, _, _, _ = model.classify(shared_x, shared_m, full_x, full_m)
                loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu().item()))

        acc, f1m = eval_classifier(model, te_loader, device)
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            print(f"[Few-shot k={kshot}] epoch {ep:03d}/{epochs} loss={np.mean(losses):.4f} acc={acc:.4f} macroF1={f1m:.4f}", flush=True)

    return eval_classifier(model, te_loader, device)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, required=True, choices=[
        "pretrain_source",
        "train_source",
        "eval_target_zeroshot",
        "eval_target_fewshot",
    ])

    parser.add_argument("--source_xlsx", type=str, default="data.xlsx")
    parser.add_argument("--target_xlsx", type=str, default="target.xlsx")

    parser.add_argument("--normalize", type=str, default="zscore", choices=["zscore", "minmax", "none"])

    parser.add_argument("--shared_min", type=float, default=1400.320416)
    parser.add_argument("--shared_max", type=float, default=4000.364384)
    parser.add_argument("--shared_len", type=int, default=1600)

    parser.add_argument("--full_min", type=float, default=400.0)
    parser.add_argument("--full_max", type=float, default=4000.364384)
    parser.add_argument("--full_len", type=int, default=2048)

    parser.add_argument("--shared_patch", type=int, default=16)
    parser.add_argument("--full_patch", type=int, default=16)

    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--max_span", type=int, default=8)
    parser.add_argument("--lambda_der", type=float, default=0.2)
    parser.add_argument("--lambda_br", type=float, default=0.5)

    parser.add_argument("--shared_dim", type=int, default=256)
    parser.add_argument("--full_dim", type=int, default=256)
    parser.add_argument("--shared_depth", type=int, default=6)
    parser.add_argument("--full_depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=8)

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-4)
    parser.add_argument("--pretrain_epochs", type=int, default=50)
    parser.add_argument("--sup_epochs", type=int, default=40)
    parser.add_argument("--fewshot_epochs", type=int, default=60)
    parser.add_argument("--kshot", type=int, default=5)

    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pretrain_ckpt", type=str, default="checkpoints/dual_pretrain.pt")
    parser.add_argument("--source_cls_ckpt", type=str, default="checkpoints/dual_source_cls.pt")

    args = parser.parse_args()
    set_seed(args.seed)

    device = require_device(args.gpu)
    use_amp = args.use_amp or (device == "cuda")
    print(f"Device: {device} | AMP: {use_amp}", flush=True)

    shared_range = (args.shared_min, args.shared_max)
    full_range = (args.full_min, args.full_max)

    source_ds = DualBranchSpectraDataset(
        xlsx_path=args.source_xlsx,
        normalize=args.normalize,
        shared_range=shared_range,
        shared_len=args.shared_len,
        full_range=full_range,
        full_len=args.full_len,
    )

    source_num_classes = len(source_ds.raw_to_contig)
    print(f"Source samples={len(source_ds)} classes={source_num_classes}", flush=True)

    model = DualBranchModel(
        shared_len=args.shared_len,
        full_len=args.full_len,
        shared_patch=args.shared_patch,
        full_patch=args.full_patch,
        shared_dim=args.shared_dim,
        full_dim=args.full_dim,
        shared_depth=args.shared_depth,
        full_depth=args.full_depth,
        heads=args.heads,
        num_classes=source_num_classes,
        fusion_dim=256,
    )

    idx = np.arange(len(source_ds))
    tr_idx, va_idx = train_test_split(idx, test_size=0.2, random_state=args.seed, stratify=source_ds.y)
    src_train = Subset(source_ds, tr_idx.tolist())
    src_val = Subset(source_ds, va_idx.tolist())

    src_train_loader = DataLoader(src_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2)
    src_val_loader = DataLoader(src_val, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if args.mode == "pretrain_source":
        cfg = TrainConfig(
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.wd,
            epochs=args.pretrain_epochs,
            num_workers=2,
            use_amp=use_amp,
            device=device,
        )
        pretrain_source(
            model=model,
            loader=src_train_loader,
            cfg=cfg,
            mask_ratio=args.mask_ratio,
            max_span=args.max_span,
            lambda_der=args.lambda_der,
            lambda_br=args.lambda_br,
            save_path=args.pretrain_ckpt,
        )
        return

    if not os.path.exists(args.pretrain_ckpt):
        raise FileNotFoundError(f"Pretrain ckpt not found: {args.pretrain_ckpt}")
    model.load_state_dict(torch.load(args.pretrain_ckpt, map_location="cpu")["model"], strict=False)
    print(f"Loaded pretrain ckpt: {args.pretrain_ckpt}", flush=True)

    if args.mode == "train_source":
        cfg = TrainConfig(
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.wd,
            epochs=args.sup_epochs,
            num_workers=2,
            use_amp=use_amp,
            device=device,
        )
        best = train_source_supervised(
            model=model,
            train_loader=src_train_loader,
            val_loader=src_val_loader,
            cfg=cfg,
            freeze_encoders=args.freeze_backbone,
            save_path=args.source_cls_ckpt,
        )
        print(f"Best source val acc: {best:.4f}", flush=True)
        return

    if not os.path.exists(args.source_cls_ckpt):
        raise FileNotFoundError(f"Source classifier ckpt not found: {args.source_cls_ckpt}")
    model.load_state_dict(torch.load(args.source_cls_ckpt, map_location="cpu")["model"], strict=False)
    print(f"Loaded source cls ckpt: {args.source_cls_ckpt}", flush=True)

    target_ds = DualBranchSpectraDataset(
        xlsx_path=args.target_xlsx,
        normalize=args.normalize,
        shared_range=shared_range,
        shared_len=args.shared_len,
        full_range=full_range,
        full_len=args.full_len,
    )
    print(f"Target samples={len(target_ds)} classes={len(target_ds.raw_to_contig)}", flush=True)

    if args.mode == "eval_target_zeroshot":
        eval_target_zeroshot(model, source_ds, target_ds, device=device, batch_size=args.batch_size)
        return

    if args.mode == "eval_target_fewshot":
        target_model = expand_classifier_to_target(model, source_ds, target_ds)

        print("\n[Stage 1] Freeze backbone, train classifier only", flush=True)
        acc1, f11 = fewshot_adapt(
            model=target_model,
            target_ds=target_ds,
            device=device,
            kshot=args.kshot,
            batch_size=min(args.batch_size, 64),
            freeze_backbone=True,
            epochs=args.fewshot_epochs,
            lr=1e-3,
            wd=args.wd,
            use_amp=use_amp,
        )

        print("\n[Stage 2] Unfreeze full model, micro fine-tune", flush=True)
        acc2, f12 = fewshot_adapt(
            model=target_model,
            target_ds=target_ds,
            device=device,
            kshot=args.kshot,
            batch_size=min(args.batch_size, 64),
            freeze_backbone=False,
            epochs=max(20, args.fewshot_epochs // 2),
            lr=1e-4,
            wd=args.wd,
            use_amp=use_amp,
        )

        print(f"[Few-shot Final] stage1_acc={acc1:.4f} stage1_f1={f11:.4f} | stage2_acc={acc2:.4f} stage2_f1={f12:.4f}", flush=True)
        return


if __name__ == "__main__":
    main()
