"""Harvest pre-MLP / MLP-out residuals from frozen Genterp, train a CLT, verify recon + sparsity."""

from __future__ import annotations

import torch

from genterp import CLTConfig, CrossLayerTranscoder, Genterp, harvest_transcoder_acts
from genterp.transcoder import _JumpReLU
from tests._factories import make_batch, tiny_config


def test_clt_training():
    torch.manual_seed(0)
    cfg = tiny_config(n_layers=3)
    base = Genterp(cfg)
    for p in base.parameters():
        p.requires_grad_(False)

    batch = make_batch(B=4, T=24, n_atoms=cfg.n_atoms)
    pre_mlp, mlp_out = harvest_transcoder_acts(base, batch)
    n_tokens, n_layers, dim = pre_mlp.shape
    assert mlp_out.shape == pre_mlp.shape
    assert n_layers == cfg.n_layers and dim == cfg.dim
    assert n_tokens > 0

    clt = CrossLayerTranscoder(CLTConfig(n_layers=n_layers, dim=dim, n_features=128, sparsity_coef=1e-3))
    opt = torch.optim.Adam(clt.parameters(), lr=1e-2)

    init_recon = clt.loss(pre_mlp, mlp_out)["recon"].item()
    for _ in range(100):
        out = clt.loss(pre_mlp, mlp_out)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    final = clt.loss(pre_mlp, mlp_out)
    assert final["recon"].item() < init_recon
    assert 0 <= final["n_active"].item() < clt.cfg.n_features


def test_clt_cross_layer_decoder_mask():
    """Decoder weights below the diagonal (target < source) must stay zero — no backward flow."""
    torch.manual_seed(0)
    clt = CrossLayerTranscoder(CLTConfig(n_layers=4, dim=8, n_features=16))
    x = torch.randn(5, 4, 8)
    target = torch.randn(5, 4, 8)
    out = clt.loss(x, target)
    out["loss"].backward()
    grad = clt.dec_weight.grad
    L = clt.cfg.n_layers
    for s in range(L):
        for t in range(s):
            assert grad[s, t].abs().max().item() == 0.0


def test_clt_sparsity_tanh_uses_same_layer_decoder_contribution_scale():
    clt = CrossLayerTranscoder(CLTConfig(n_layers=1, dim=2, n_features=2))
    features = torch.tensor([[[10.0, 1.0]]])

    with torch.no_grad():
        clt.per_layer_std.fill_(2.0)
        clt.dec_weight.zero_()
        clt.dec_weight[0, 0, 0] = torch.tensor([0.2, 0.0])
        clt.dec_weight[0, 0, 1] = torch.tensor([0.0, 2.0])

    sparsity = clt._sparsity(features)

    assert torch.allclose(sparsity, 2 * torch.tanh(torch.tensor(1.0)))


def test_clt_recon_loss_normalizes_per_layer_activation_scale():
    clt = CrossLayerTranscoder(CLTConfig(n_layers=2, dim=2, n_features=4))
    pre_mlp = torch.zeros(2, 2, 2)
    mlp_out = torch.tensor(
        [
            [[1.0, -1.0], [10.0, -10.0]],
            [[1.0, -1.0], [10.0, -10.0]],
        ]
    )

    out = clt.loss(pre_mlp, mlp_out)

    assert torch.allclose(clt.per_layer_std, torch.tensor([1.0, 10.0]))
    assert torch.allclose(out["recon"], torch.tensor(2.0))


def test_jumprelu_threshold_ste_bandwidth_scales_with_threshold():
    x = torch.tensor([[1.09, 10.9], [1.11, 11.1]])
    theta = torch.tensor([1.0, 10.0], requires_grad=True)
    bandwidth = theta.detach() * 0.2

    _JumpReLU.apply(x, theta, bandwidth).sum().backward()

    assert torch.allclose(theta.grad, torch.tensor([-5.0, -5.0]))


def test_clt_eval_loss_does_not_update_per_layer_activation_scale():
    clt = CrossLayerTranscoder(CLTConfig(n_layers=2, dim=2, n_features=4))
    pre_mlp = torch.zeros(2, 2, 2)
    train_mlp_out = torch.tensor(
        [
            [[1.0, -1.0], [10.0, -10.0]],
            [[1.0, -1.0], [10.0, -10.0]],
        ]
    )
    eval_mlp_out = torch.tensor(
        [
            [[100.0, -100.0], [2.0, -2.0]],
            [[100.0, -100.0], [2.0, -2.0]],
        ]
    )

    clt.loss(pre_mlp, train_mlp_out)
    before = clt.per_layer_std.clone()
    initialized_before = clt.per_layer_std_initialized.clone()

    clt.eval()
    clt.loss(pre_mlp, eval_mlp_out)

    assert torch.equal(clt.per_layer_std_initialized, initialized_before)
    assert torch.allclose(clt.per_layer_std, before)


def test_clt_eval_loss_does_not_initialize_per_layer_activation_scale():
    clt = CrossLayerTranscoder(CLTConfig(n_layers=2, dim=2, n_features=4)).eval()
    pre_mlp = torch.zeros(2, 2, 2)
    mlp_out = torch.tensor(
        [
            [[1.0, -1.0], [10.0, -10.0]],
            [[1.0, -1.0], [10.0, -10.0]],
        ]
    )

    clt.loss(pre_mlp, mlp_out)

    assert not clt.per_layer_std_initialized.item()
    assert torch.allclose(clt.per_layer_std, torch.ones(2))
