"""
pipeline.py
───────────
三阶段光谱学习主流水线（累积采样版）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 必须手动指定的参数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --data       预训练数据文件路径（data.xlsx）
  --new_batch  新批次文件路径
  --save_dir   权重 & 采样顺序保存目录（默认 checkpoints/）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 累积采样保证
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1-shot ⊂ 5-shot ⊂ 10-shot ⊂ 20-shot
  首次运行时为每类生成固定随机排列，存入 save_dir/。
  后续不同 shots 的实验复用同一排列，只取前 N 个。

  推荐执行顺序（与 shots 大小无关，顺序均正确）：

    echo "===== 1-shot full pipeline =====" > results_all.txt
    python pipeline.py \\
        --data data.xlsx --new_batch dataceshi1600.xlsx \\
        --save_dir exp_autoencoderyanshi \\
        --shots 1 >> results_all.txt 2>&1

    echo "===== 5-shot stage3 only =====" >> results_all.txt
    python pipeline.py \\
        --data data.xlsx --new_batch dataceshi1600.xlsx \\
        --save_dir exp_autoencoderyanshi \\
        --stage 3 --shots 5 >> results_all.txt 2>&1

    echo "===== 10-shot stage3 only =====" >> results_all.txt
    python pipeline.py \\
        --data data.xlsx --new_batch dataceshi1600.xlsx \\
        --save_dir exp_autoencoderyanshi \\
        --stage 3 --shots 10 >> results_all.txt 2>&1

    echo "===== 20-shot stage3 only =====" >> results_all.txt
    python pipeline.py \\
        --data data.xlsx --new_batch dataceshi1600.xlsx \\
        --save_dir exp_autoencoderyanshi \\
        --stage 3 --shots 20 >> results_all.txt 2>&1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 可选 stage3_mode
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  linear | finetune | prototype | linear_weighted |
  linear_lbfgs | linear_norm | svm | all（默认）
"""

import os
import sys
import argparse
import torch
from pathlib import Path

from data_utils          import load_excel, split_pretrain_newbatch
from models              import Encoder1D
from stage1_pretrain     import run_stage1
from stage2_adapt        import run_stage2
from stage3_fewshot3     import run_stage3
from cumulative_sampler  import cumulative_sample   # ← 新增


# ──────────────────────────────────────────────────────────────
ENCODER_OUT_DIM = 128
ENCODER_HIDDEN  = 256
TARGET_LEN      = 256   # 0 = 保持原始列数


# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="三阶段光谱学习流水线（累积采样版）"
    )

    g_data = p.add_argument_group("数据文件（必须手动指定）")
    g_data.add_argument("--data",      required=True, metavar="FILE")
    g_data.add_argument("--new_batch", default=None,  metavar="FILE")

    g_ckpt = p.add_argument_group("权重保存")
    g_ckpt.add_argument("--save_dir", default="checkpoints", metavar="DIR")

    g_flow = p.add_argument_group("流程控制")
    g_flow.add_argument("--stage", type=int, default=0,
                        help="0=全流程  1=仅Stage1  2=仅Stage2  3=仅Stage3")

    g1 = p.add_argument_group("Stage 1 预训练")
    g1.add_argument("--stage1_mode",   default="supervised")
    g1.add_argument("--stage1_epochs", type=int,   default=60)
    g1.add_argument("--stage1_lr",     type=float, default=1e-3)
    g1.add_argument("--stage1_batch",  type=int,   default=64)

    g2 = p.add_argument_group("Stage 2 自适应")
    g2.add_argument("--stage2_mode",   default="all")
    g2.add_argument("--stage2_epochs", type=int,   default=30)
    g2.add_argument("--stage2_lr",     type=float, default=5e-4)
    g2.add_argument("--stage2_batch",  type=int,   default=32)

    g3 = p.add_argument_group("Stage 3 Few-Shot 分类")
    g3.add_argument("--shots",         type=int,   default=5,
                    help="每类支持样本数：1 / 5 / 10 / 20 ...")
    g3.add_argument("--stage3_mode",   default="all")
    g3.add_argument("--stage3_epochs", type=int,   default=60)
    g3.add_argument("--stage3_lr",     type=float, default=1e-3)

    # ── 累积采样参数 ──────────────────────────────────────────
    g_cs = p.add_argument_group("累积采样（保证 1⊂5⊂10⊂20-shot）")
    g_cs.add_argument("--sample_seed",    type=int,   default=42,
                      help="首次生成固定排列时的随机种子（默认 42）")
    g_cs.add_argument("--query_ratio",    type=float, default=0.4,
                      help="查询集占非支持集的比例（首次固定，后续复用）")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_len = TARGET_LEN if TARGET_LEN > 0 else None

    print(f"\n{'='*65}")
    print(f"  三阶段光谱学习流水线（累积采样版）")
    print(f"  Device    : {device}")
    print(f"  save_dir  : {Path(args.save_dir).resolve()}")
    print(f"{'='*65}")

    # ── 加载预训练数据 ────────────────────────────────────────
    print(f"\n[数据] 预训练集: {args.data}")
    if not Path(args.data).exists():
        print(f"  ✗ 找不到文件: {args.data}"); sys.exit(1)
    X_pre, y_pre, class_names, signal_len = load_excel(args.data, target_len)
    n_classes = len(class_names)
    print(f"  samples={len(X_pre)}  signal_len={signal_len}  "
          f"n_classes={n_classes}  classes={class_names}")

    # ── 加载新批次数据 ────────────────────────────────────────
    new_batch_path = args.new_batch
    if new_batch_path and new_batch_path not in ("", "none", "None"):
        if not Path(new_batch_path).exists():
            print(f"  ✗ 找不到新批次文件: {new_batch_path}"); sys.exit(1)
        print(f"\n[数据] 新批次文件: {new_batch_path}")
        new_X, new_y, new_classes, _ = load_excel(new_batch_path, target_len)
        n_new_classes  = len(new_classes)
        pretrain_X, pretrain_y = X_pre, y_pre
        print(f"  samples={len(new_X)}  n_classes={n_new_classes}  "
              f"classes={new_classes}")
    else:
        print(f"\n[数据] 未指定 --new_batch，从 {args.data} 内部切出 20%")
        pretrain_X, pretrain_y, new_X, new_y = split_pretrain_newbatch(
            X_pre, y_pre, new_batch_ratio=0.2)
        n_new_classes  = n_classes
        new_batch_path = None
        print(f"  pretrain={len(pretrain_X)}  new_batch={len(new_X)}")

    # ── 构建 Encoder ──────────────────────────────────────────
    encoder = Encoder1D(
        signal_len=signal_len,
        out_dim=ENCODER_OUT_DIM,
        hidden_dim=ENCODER_HIDDEN,
    ).to(device)
    print(f"\n  Encoder 参数量: {sum(p.numel() for p in encoder.parameters()):,}")

    ckpt1 = Path(args.save_dir) / "stage1_encoder.pt"
    ckpt2 = Path(args.save_dir) / "stage2_encoder.pt"

    # ══════════════════════════════════════════════════════════
    #  Stage 1
    # ══════════════════════════════════════════════════════════
    if args.stage in (0, 1):
        print(f"\n{'='*65}")
        print(f"  Stage 1：通用预训练  [{args.stage1_mode}]")
        print(f"{'='*65}")
        run_stage1(
            encoder=encoder, X=pretrain_X, y=pretrain_y,
            n_classes=n_classes, signal_len=signal_len,
            mode=args.stage1_mode,
            epochs=args.stage1_epochs, lr=args.stage1_lr,
            batch_size=args.stage1_batch,
            device=device, save_path=str(ckpt1),
        )

    # ══════════════════════════════════════════════════════════
    #  Stage 2
    # ══════════════════════════════════════════════════════════
    if args.stage in (0, 2):
        if ckpt1.exists():
            encoder.load_state_dict(torch.load(ckpt1, map_location=device))
            print(f"\n  ✔ 已加载 Stage1 权重：{ckpt1}")
        else:
            print(f"\n  ⚠ Stage1 权重不存在，使用随机初始化")
        src = new_batch_path or f"{args.data}（内部切分）"
        print(f"\n{'='*65}")
        print(f"  Stage 2：新批次自监督预适应  [{args.stage2_mode}]")
        print(f"  文件 → {src}")
        print(f"{'='*65}")
        run_stage2(
            encoder=encoder, X_new=new_X, signal_len=signal_len,
            mode=args.stage2_mode,
            epochs=args.stage2_epochs, lr=args.stage2_lr,
            batch_size=args.stage2_batch,
            device=device, save_path=str(ckpt2),
        )

    # ══════════════════════════════════════════════════════════
    #  Stage 3：Few-Shot（使用累积采样）
    # ══════════════════════════════════════════════════════════
    if args.stage in (0, 3):
        # 加载权重
        loaded = False
        for ckpt in (ckpt2, ckpt1):
            if ckpt.exists():
                encoder.load_state_dict(torch.load(ckpt, map_location=device))
                print(f"\n  ✔ 已加载权重：{ckpt}")
                loaded = True
                break
        if not loaded:
            print(f"\n  ⚠ 未找到任何已保存权重，使用随机初始化 encoder")

        src = new_batch_path or f"{args.data}（内部切分）"
        print(f"\n{'='*65}")
        print(f"  Stage 3：Few-Shot 分类（固定累积采样）")
        print(f"  文件   → {src}")
        print(f"  shots  → {args.shots}/class  mode → {args.stage3_mode}")
        print(f"  epochs → {args.stage3_epochs}  lr={args.stage3_lr}")
        print(f"  seed   → {args.sample_seed}  query_ratio={args.query_ratio}")
        print(f"{'='*65}")

        # ── 累积采样：保证 1⊂5⊂10⊂20-shot ──────────────────
        #   固定排列文件存到 save_dir/，跨 shots 复用
        X_sup, y_sup, X_qry, y_qry = cumulative_sample(
            X           = new_X,
            y           = new_y,
            shots_per_class = args.shots,
            save_dir    = args.save_dir,        # 复用同一目录
            seed        = args.sample_seed,
            query_ratio = args.query_ratio,
        )

        run_stage3(
            encoder         = encoder,
            X_new           = new_X,            # 兼容性保留，实际不再内部采样
            y_new           = new_y,
            n_classes       = n_new_classes,
            shots_per_class = args.shots,
            mode            = args.stage3_mode,
            encoder_dim     = ENCODER_OUT_DIM,
            epochs          = args.stage3_epochs,
            lr              = args.stage3_lr,
            device          = device,
            save_dir        = args.save_dir,
            # ↓ 传入预切分数据，跳过内部随机采样
            X_sup=X_sup, y_sup=y_sup,
            X_qry=X_qry, y_qry=y_qry,
        )

    print(f"\n{'='*65}")
    print(f"  ✅  完成！  权重目录 → {Path(args.save_dir).resolve()}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
