"""Basic CPU checks for the dual-branch models and observation masks."""
import importlib
import numpy as np
import torch


def check(module_name):
    module = importlib.import_module(module_name)
    module.set_seed(42)
    values, mask = module.linear_resample_with_mask(
        np.array([2., 1., 0.]), np.array([2., 1., 0.]), -1., 3., 5
    )
    np.testing.assert_array_equal(mask, [0., 1., 1., 1., 0.])
    np.testing.assert_allclose(values, [0., 0., 1., 2., 0.])
    model = module.DualBranchModel(
        shared_len=32, full_len=48, shared_patch=8, full_patch=8,
        shared_dim=16, full_dim=16, shared_depth=1, full_depth=1,
        heads=4, num_classes=3, fusion_dim=12, dropout=0.
    )
    sx, fx = torch.randn(2, 32), torch.randn(2, 48)
    sm, fm = torch.ones_like(sx), torch.ones_like(fx)
    fm[:, :8] = 0.
    fx[:, :8] = 0.
    logits, zs, zf, z = model.classify(sx, sm, fx, fm)
    assert logits.shape == (2, 3)
    assert zs.shape == zf.shape == (2, 16)
    assert z.shape == (2, 12)
    pred, masked = model.reconstruct_shared(sx, sm, mask_ratio=.5)
    assert pred.shape == (2, 4, 8)
    assert masked.shape == (2, 4) and masked.any()
    loss = logits.square().mean() + pred.square().mean()
    loss.backward()
    for encoder in (model.shared_encoder, model.full_encoder):
        assert encoder.patch_embed_x.weight.grad is not None
        assert encoder.patch_embed_m.weight.grad is not None
    assert torch.isfinite(loss)
    print(module_name + ': passed')


if __name__ == '__main__':
    check('spectral_dualbranch_transfer')
    check('spectral_dualbranch_transfer1')
