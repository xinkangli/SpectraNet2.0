# SpectraNet2.0

Dual-branch Transformer implementations for FTIR representation learning. The shared-range encoder and the full-spectrum encoder process spectral intensities together with observation-validity masks. Their pooled representations are concatenated and fused by an MLP.

## Files

- `spectral_dualbranch_transfer.py`: dual-branch source pretraining, supervised source training, zero-shot evaluation, and few-shot adaptation.
- `spectral_dualbranch_transfer1.py`: an extended implementation with target-inclusive pretraining, spectral augmentation, MMD alignment, optional domain-adversarial alignment, and prototype evaluation.
- `smoke_test.py`: checks resampling masks, both branches, feature fusion, reconstruction output, and gradient propagation using small synthetic inputs.

Both implementations are retained under their server filenames. The extended implementation is used in the commands below. Checkpoints are specific to the architecture and dimensions with which they were trained.

## Installation

Use Python 3.10 or later and install dependencies:

```bash
python -m pip install -r requirements.txt
python smoke_test.py
python spectral_dualbranch_transfer1.py --help
```

## Input data

Each Excel row contains one spectrum. Spectral column names are numeric wavenumbers in cm^-1; the last column contains an integer class label and must have a nonnumeric header, for example `label`. Spectral values within the recorded coverage must be finite. Spectra are sorted by wavenumber and linearly interpolated onto each branch grid. Positions outside the measured range are filled with zero and assigned observation mask zero; interpolated positions within coverage have mask one. This loader uses the extent of the recorded wavenumber axis and does not infer internal missing intervals from absent columns or NaN values.

Source and target integer labels must refer to the same polymer wherever labels overlap, because classifier transfer uses raw-label matching. For datasets with different numeric conventions, remap labels by polymer identity before running classification transfer.

## Architecture and defaults

| Setting | Default |
| --- | --- |
| Shared wavenumber grid | 1400.320416–4000.364384 cm^-1, 1600 points |
| Full-spectrum grid | 400–4000.364384 cm^-1, 2048 points |
| Patch size | 16 for both branches |
| Branch embedding dimension | 256 each |
| Transformer depth / heads | 6 / 8 for each branch |
| Concatenated / fused dimension | 512 / 256 |
| Decoder dimension / depth / heads | 128 / 2 / 4 |
| Normalization | Per-spectrum z-score over observed values |
| Reconstruction mask ratio | 0.6 |
| Derivative / branch loss weights | 0.2 / 0.5 |

The input tokens sum intensity-patch embeddings, observation-mask embeddings, and positional embeddings. Reconstruction uses the shared branch. The default mask ratio is a command-line default and can be set explicitly; use `--mask_ratio 0.3` for a 30% setting. The `minmax` option in these files scales observed values to [0, 1].

## Example commands

Source self-supervised pretraining:

```bash
python spectral_dualbranch_transfer1.py --mode pretrain_source --source_xlsx data.xlsx --pretrain_epochs 50 --mask_ratio 0.3 --pretrain_ckpt checkpoints/dual_pretrain.pt
```

Supervised source classifier training:

```bash
python spectral_dualbranch_transfer1.py --mode train_source --source_xlsx data.xlsx --pretrain_ckpt checkpoints/dual_pretrain.pt --source_cls_ckpt checkpoints/dual_source_cls.pt
```

Target-inclusive pretraining with MMD alignment:

```bash
python spectral_dualbranch_transfer1.py --mode pretrain_with_target --source_xlsx data.xlsx --target_xlsx target.xlsx --mask_ratio 0.3 --lambda_mmd 1.0 --lambda_dann 0.0 --pretrain_da_ckpt checkpoints/dual_pretrain_da.pt
```

Prototype evaluation:

```bash
python spectral_dualbranch_transfer1.py --mode eval_prototype --source_xlsx data.xlsx --target_xlsx target.xlsx --source_cls_ckpt checkpoints/dual_source_cls.pt --kshot 5 --seed 42
```

The extended script prefers an existing `pretrain_da_ckpt` over `pretrain_ckpt` when loading. Use separate checkpoint directories for different settings to control this selection. Target-inclusive pretraining accesses every spectrum passed in `target_xlsx`. For an evaluation that excludes queries from adaptation, pass support-only target data during adaptation and keep held-out queries separate. The built-in few-shot routine samples records within each class; specimen provenance and synthetic-parent separation must be established before using that routine. The commands demonstrate available modes; experiment-specific results require their corresponding data, checkpoints, and recorded settings.

## Model code locations

`PatchTransformerEncoder` defines intensity and observation-mask embeddings. `DualBranchModel` defines the two encoders, shared decoder, fusion MLP, and classification head. `linear_resample_with_mask` performs coordinate-based interpolation and observation-mask construction. `cosine_consistency` implements the cross-branch representation objective.
