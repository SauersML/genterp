"""Harvest pre-MLP / MLP-out residuals from frozen Genterp, train a CLT, verify recon + sparsity."""

from __future__ import annotations

import torch

from genterp import (
    CLTConfig,
    CrossLayerTranscoder,
    Genterp,
    feature_to_feature_attribution_graph,
    feature_to_output_attribution,
    harvest_transcoder_acts,
    top_activating_examples,
    unwrap_genterp_model,
)
from genterp.transcoder import _JumpReLU
from tests._factories import make_batch, tiny_config


class _SavedModelWrapper(torch.nn.Module):
    def __init__(self, model: Genterp):
        super().__init__()
        self.model = model


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


def test_harvest_transcoder_acts_accepts_saved_training_wrapper():
    cfg = tiny_config(n_layers=2)
    wrapped = _SavedModelWrapper(Genterp(cfg))
    wrapped.train()
    batch = make_batch(B=2, T=8, n_atoms=cfg.n_atoms)

    assert unwrap_genterp_model(wrapped) is wrapped.model
    pre_mlp, mlp_out = harvest_transcoder_acts(wrapped, batch)

    assert wrapped.model.training
    assert pre_mlp.shape == mlp_out.shape
    assert pre_mlp.shape[1:] == (cfg.n_layers, cfg.dim)
    assert pre_mlp.shape[0] > 0


def test_clt_cross_layer_decoder_mask():
    """Decoder only stores same-layer and strict future-layer writes."""
    torch.manual_seed(0)
    clt = CrossLayerTranscoder(CLTConfig(n_layers=4, dim=8, n_features=16))
    x = torch.randn(5, 4, 8)
    target = torch.randn(5, 4, 8)
    out = clt.loss(x, target)
    out["loss"].backward()

    assert all(s < t for s, t in zip(clt._off_s_idx.tolist(), clt._off_t_idx.tolist(), strict=True))
    assert clt.diag_W.grad is not None
    assert clt.off_blocks is not None
    assert clt.off_blocks.grad is not None


def test_clt_low_rank_off_diagonal_storage_shapes():
    """When off_diagonal_rank is set, off-diagonals live in U (per pair) and V (per source)."""
    L, F, D, r = 4, 16, 8, 3
    clt = CrossLayerTranscoder(
        CLTConfig(n_layers=L, dim=D, n_features=F, off_diagonal_rank=r)
    )
    n_off = L * (L - 1) // 2

    assert clt.off_blocks is None
    assert clt.diag_W.shape == (L, F, D)
    assert clt.off_U is not None and clt.off_U.shape == (n_off, F, r)
    assert clt.off_V is not None and clt.off_V.shape == (L, r, D)


def test_clt_low_rank_decode_matches_explicit_uv_product():
    """The factorized decode path must agree with the materialized U @ V[s] block-product."""
    torch.manual_seed(0)
    L, F, D, r = 3, 7, 5, 2
    clt = CrossLayerTranscoder(
        CLTConfig(n_layers=L, dim=D, n_features=F, off_diagonal_rank=r)
    ).eval()
    features = torch.randn(4, L, F)

    with torch.no_grad():
        out = clt.decode(features)

    assert clt.off_U is not None and clt.off_V is not None
    expected = torch.einsum("nlf,lfd->nld", features, clt.diag_W) + clt.dec_bias
    for pair_idx, (s, t) in enumerate(zip(clt._off_s_idx.tolist(), clt._off_t_idx.tolist(), strict=True)):
        block = clt.off_U[pair_idx] @ clt.off_V[s]
        expected[:, t, :] = expected[:, t, :] + features[:, s, :] @ block
    assert torch.allclose(out, expected, atol=1e-5)


def test_clt_low_rank_trains_and_lowers_recon_loss():
    """A factorized CLT should still optimize: param count drops, learning still works."""
    torch.manual_seed(0)
    L, F, D, r = 3, 32, 8, 4
    clt = CrossLayerTranscoder(
        CLTConfig(n_layers=L, dim=D, n_features=F, off_diagonal_rank=r, sparsity_coef=1e-3)
    )
    pre_mlp = torch.randn(16, L, D)
    mlp_out = torch.randn(16, L, D)
    opt = torch.optim.Adam(clt.parameters(), lr=5e-2)

    init_recon = clt.loss(pre_mlp, mlp_out)["recon"].item()
    for _ in range(100):
        out = clt.loss(pre_mlp, mlp_out)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
    assert clt.loss(pre_mlp, mlp_out)["recon"].item() < init_recon


def test_clt_sparsity_tanh_uses_same_layer_decoder_contribution_scale():
    clt = CrossLayerTranscoder(CLTConfig(n_layers=1, dim=2, n_features=2))
    features = torch.tensor([[[10.0, 1.0]]])

    with torch.no_grad():
        clt.per_layer_std.fill_(2.0)
        clt.diag_W.zero_()
        clt.diag_W[0, 0] = torch.tensor([0.2, 0.0])
        clt.diag_W[0, 1] = torch.tensor([0.0, 2.0])

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


def test_top_activating_examples_returns_event_windows():
    cfg = tiny_config(n_layers=2)
    model = Genterp(cfg).eval()
    batch = make_batch(B=2, T=8, n_atoms=cfg.n_atoms)
    batch["event_pad"][1, -2:] = True
    clt = CrossLayerTranscoder(CLTConfig(n_layers=cfg.n_layers, dim=cfg.dim, n_features=8)).eval()

    windows = top_activating_examples(model, clt, batch, layer=0, feature=0, k=3, window_radius=1)

    assert len(windows) == 3
    assert all(window.layer == 0 and window.feature == 0 for window in windows)
    assert all(window.event_atoms.numel() <= 3 for window in windows)
    assert all(not batch["event_pad"][window.batch_index, window.token_index] for window in windows)


def test_feature_to_output_attribution_shapes():
    cfg = tiny_config(n_atoms=32, dim=16, n_heads=4, n_layers=2)
    model = Genterp(cfg).eval()
    clt = CrossLayerTranscoder(CLTConfig(n_layers=cfg.n_layers, dim=cfg.dim, n_features=5)).eval()
    n_tokens = 4
    pre_mlp = torch.randn(n_tokens, cfg.n_layers, cfg.dim)
    hidden = torch.randn(n_tokens, cfg.dim)
    delta_t = torch.ones(n_tokens)

    out = feature_to_output_attribution(
        clt,
        model.tpp,
        hidden,
        pre_mlp,
        delta_t,
        mark_indices=[1, 2],
        time_indices=[0, 1, 2],
    )

    assert out["features"].shape == (n_tokens, cfg.n_layers, clt.cfg.n_features)
    assert out["mark_grad"].shape == (n_tokens, cfg.n_layers, clt.cfg.n_features, 2)
    assert out["time_grad"].shape == (n_tokens, cfg.n_layers, clt.cfg.n_features, 3)
    assert out["mark_activation_attribution"].shape == out["mark_grad"].shape
    assert out["time_activation_attribution"].shape == out["time_grad"].shape


def test_feature_to_output_attribution_matches_autograd():
    cfg = tiny_config(n_atoms=32, dim=16, n_heads=4, n_layers=2)
    model = Genterp(cfg).eval()
    clt = CrossLayerTranscoder(CLTConfig(n_layers=cfg.n_layers, dim=cfg.dim, n_features=5)).eval()
    pre_mlp = torch.randn(3, cfg.n_layers, cfg.dim)
    hidden = torch.randn(3, cfg.dim)
    delta_t = torch.ones(3)

    out = feature_to_output_attribution(
        clt,
        model.tpp,
        hidden,
        pre_mlp,
        delta_t,
        mark_indices=[1],
        time_indices=[0],
    )
    features = clt.encode(pre_mlp.detach()).requires_grad_(True)
    decoded = clt.decode(features)
    hidden_proxy = hidden.detach() + decoded.sum(dim=1) - decoded.detach().sum(dim=1)
    mark_grad = torch.autograd.grad(model.tpp.mark_logits(hidden_proxy, delta_t)[:, 1].sum(), features, retain_graph=True)[0]
    time_grad = torch.autograd.grad(model.tpp.time_logits(hidden_proxy)[:, 0].sum(), features)[0]

    assert torch.allclose(out["mark_grad"].squeeze(-1), mark_grad, atol=1e-6)
    assert torch.allclose(out["time_grad"].squeeze(-1), time_grad, atol=1e-6)


def test_feature_to_feature_graph_uses_decoder_to_encoder_effects():
    clt = CrossLayerTranscoder(CLTConfig(n_layers=2, dim=2, n_features=2)).eval()
    with torch.no_grad():
        clt.enc_weight.zero_()
        clt.diag_W.zero_()
        assert clt.off_blocks is not None
        clt.off_blocks.zero_()
        clt.enc_weight[1, 0] = torch.tensor([1.0, 0.0])
        clt.enc_weight[1, 1] = torch.tensor([0.0, 1.0])
        clt.off_blocks[0, 1] = torch.tensor([3.0, 4.0])

    edges = feature_to_feature_attribution_graph(clt, top_k_per_layer_pair=2)

    assert [(edge.source_layer, edge.target_layer) for edge in edges] == [(0, 1), (0, 1)]
    assert edges[0].source_feature == 1
    assert edges[0].target_feature == 1
    assert edges[0].weight == 4.0
    assert edges[1].source_feature == 1
    assert edges[1].target_feature == 0
    assert edges[1].weight == 3.0
