"""
cumulative_sampler.py
─────────────────────
固定累积采样：保证
    1-shot ⊂ 5-shot ⊂ 10-shot ⊂ 20-shot

原理
────
对每个类别 c，将其所有样本索引随机打乱一次（seed 固定），
结果保存到 <save_dir>/fewshot_order_class{c}.npy。
后续调用只从该固定顺序中取前 N 个，永远不替换已取的样本。

文件布局（均在 save_dir 下）
────────────────────────────
  fewshot_order_class{c}.npy   → 类 c 的固定随机排列索引（全量）
  fewshot_query.npy            → 查询集索引（固定，一次性生成）
  fewshot_query_labels.npy     → 查询集标签

公开接口
────────
  cumulative_sample(X, y, shots_per_class, save_dir, seed=42, query_ratio=0.4)
      → X_sup, y_sup, X_qry, y_qry

      shots_per_class : 本次实验需要的每类支持样本数
      query_ratio     : 从非支持样本中取多少比例做查询集
                        （第一次调用时固定，之后复用）
"""

import os
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────────────────────────
def cumulative_sample(X, y, shots_per_class, save_dir,
                      seed=42, query_ratio=0.4):
    """
    Parameters
    ----------
    X               : np.ndarray  (N, L)
    y               : array-like  (N,)  int 标签 0..C-1
    shots_per_class : int         本次取多少支持样本/类
    save_dir        : str / Path  固定顺序文件存放目录
    seed            : int         首次生成排列时的随机种子
    query_ratio     : float       查询集占非支持集的比例

    Returns
    -------
    X_sup, y_sup, X_qry, y_qry  均为 np.ndarray
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    y = np.asarray(y, dtype=int)
    classes = np.unique(y)
    n_classes = len(classes)

    # ── 1. 为每个类生成/读取固定随机排列 ─────────────────────
    order_per_class = {}   # class_id -> 全量索引排列
    for c in classes:
        order_file = save_dir / f"fewshot_order_class{c}.npy"
        idx_c = np.where(y == c)[0]

        if order_file.exists():
            order = np.load(order_file)
            # 校验：若数据集样本数变化，重新生成
            if len(order) != len(idx_c) or not np.array_equal(
                np.sort(order), np.sort(idx_c)
            ):
                print(f"  [累积采样] 类 {c} 数据集已变化，重新生成排列")
                order = _shuffle(idx_c, seed + int(c))
                np.save(order_file, order)
        else:
            order = _shuffle(idx_c, seed + int(c))
            np.save(order_file, order)
            print(f"  [累积采样] 类 {c}: 生成固定排列 → {order_file.name}")

        if shots_per_class > len(order):
            raise ValueError(
                f"类 {c} 共 {len(order)} 个样本，"
                f"但请求 shots_per_class={shots_per_class}，样本不足！"
            )
        order_per_class[c] = order

    # ── 2. 取支持集（前 shots_per_class 个） ─────────────────
    sup_idx = np.concatenate([
        order_per_class[c][:shots_per_class] for c in classes
    ])

    # ── 3. 生成/读取固定查询集 ────────────────────────────────
    qry_file   = save_dir / "fewshot_query.npy"
    qry_lbl_f  = save_dir / "fewshot_query_labels.npy"

    if qry_file.exists() and qry_lbl_f.exists():
        qry_idx = np.load(qry_file)
        # 查询集不能和当前支持集重叠（支持集可能比生成时更大）
        qry_idx = qry_idx[~np.isin(qry_idx, sup_idx)]
        if len(qry_idx) == 0:
            print("  [累积采样] ⚠ 查询集与支持集完全重叠，强制重新生成")
            qry_idx = _build_query(order_per_class, classes,
                                   shots_per_class, query_ratio,
                                   seed, qry_file, qry_lbl_f, y)
    else:
        qry_idx = _build_query(order_per_class, classes,
                               shots_per_class, query_ratio,
                               seed, qry_file, qry_lbl_f, y)

    # ── 4. 整理返回值 ─────────────────────────────────────────
    X_sup = X[sup_idx]
    y_sup = y[sup_idx]
    X_qry = X[qry_idx]
    y_qry = y[qry_idx]

    _print_info(y_sup, y_qry, classes, shots_per_class)
    return X_sup, y_sup, X_qry, y_qry


# ─────────────────────────────────────────────────────────────
#  内部工具
# ─────────────────────────────────────────────────────────────

def _shuffle(idx_array, seed):
    rng = np.random.default_rng(seed)
    return rng.permutation(idx_array)


def _build_query(order_per_class, classes, shots_per_class,
                 query_ratio, seed, qry_file, qry_lbl_f, y):
    """
    从每类的"非支持集"部分按 query_ratio 采样查询集，
    并写入磁盘（只在第一次调用时执行）。
    注意：查询集基于 shots_per_class=1（最小支持集）的非支持部分，
    确保之后增大 shots 时不污染查询集。
    """
    # 基于 shots=1 的剩余部分来构建查询集，保证最严格的不重叠
    qry_candidates = np.concatenate([
        order_per_class[c][1:] for c in classes   # 跳过第 0 个（1-shot 支持）
    ])
    rng = np.random.default_rng(seed + 9999)
    n_qry = max(1, int(len(qry_candidates) * query_ratio))
    qry_idx = rng.choice(qry_candidates, size=n_qry, replace=False)

    np.save(qry_file,  qry_idx)
    np.save(qry_lbl_f, y[qry_idx])
    print(f"  [累积采样] 查询集已固定：{len(qry_idx)} 条 → {qry_file.name}")
    return qry_idx


def _print_info(y_sup, y_qry, classes, shots):
    counts = {c: int((y_sup == c).sum()) for c in classes}
    print(f"  [累积采样] shots={shots}/class  "
          f"支持集={len(y_sup)}  查询集={len(y_qry)}")
    print(f"  支持集分布: { {c: counts[c] for c in classes} }")
