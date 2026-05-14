"""Forward + backward through Genterp with joint TPP + value loss; dict return; transcoder acts; sampling."""

from __future__ import annotations

import torch

from genterp import Genterp
from genterp.modeling import MarkedTPPHead, marked_tpp_value_loss
from tests._factories import make_batch, tiny_config


def _mark_some_atoms_magnitude(model: Genterp, frac: float = 0.5) -> None:
    """Pretend a fraction of atoms are magnitude-bearing so the value pathway gets exercised."""
    n = model.cfg.n_atoms
    mask = torch.zeros(n, dtype=torch.bool)
    mask[torch.randperm(n)[: int(frac * n)]] = True
    mask[0] = False
    model.value_mod.set_stats(
        value_mu=torch.zeros(n),
        value_sigma=torch.ones(n),
        atom_has_mag=mask,
    )


def test_forward_backward():
    cfg = tiny_config()
    model = Genterp(cfg)
    _mark_some_atoms_magnitude(model)
    batch = make_batch(n_atoms=cfg.n_atoms)

    out = model(**batch)
    B, T = batch["event_ages"].shape
    assert out["hidden"].shape == (B, T, cfg.dim)

    ld = model.loss(**batch)
    assert torch.isfinite(ld["loss"])
    assert ld["n_mag"].item() > 0
    ld["loss"].backward()

    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
    assert grad_norm > 0


def test_mark_output_uses_atom_embedding_weight():
    model = Genterp(tiny_config())

    assert model.tpp.mark_out.weight is model.embed.embedding.weight


def test_forward_uses_sex_context():
    cfg = tiny_config()
    torch.manual_seed(0)
    model = Genterp(cfg).eval()
    batch = make_batch(B=2, n_atoms=cfg.n_atoms)
    same_inputs = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    same_inputs["sex"] = torch.zeros_like(batch["sex"])
    changed_sex = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in same_inputs.items()}
    changed_sex["sex"] = torch.ones_like(batch["sex"])

    with torch.no_grad():
        out_a = model(**same_inputs)["hidden"]
        out_b = model(**changed_sex)["hidden"]

    assert not torch.allclose(out_a, out_b)


def test_value_head_student_t_limits_outlier_gradients():
    cfg = tiny_config()
    model = Genterp(cfg)
    hidden = torch.zeros(1, cfg.dim, requires_grad=True)
    concept = torch.zeros(1, cfg.dim)
    target = torch.tensor([100.0])

    loss = model.value_head.nll(hidden, concept, target).sum()
    loss.backward()

    assert torch.isfinite(loss)
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad.norm().item() < 10.0


def test_value_loss_reports_and_clips_extreme_magnitudes():
    cfg = tiny_config()
    model = Genterp(cfg)
    model.value_mod.set_stats(
        value_mu=torch.zeros(cfg.n_atoms),
        value_sigma=torch.ones(cfg.n_atoms),
        atom_has_mag=torch.ones(cfg.n_atoms, dtype=torch.bool),
    )
    batch = make_batch(n_atoms=cfg.n_atoms)
    batch["event_values"] = torch.full_like(batch["event_values"], 1e30)

    ld = model.loss(**batch)

    assert torch.isfinite(ld["loss"])
    assert ld["value_z_abs_max"].item() <= model.value_mod.z_clip
    assert ld["value_z_clipped"].item() > 0
    assert ld["value_nll_max"].item() < 20.0


def test_mark_negative_cache_is_transient():
    weight = torch.nn.Parameter(torch.randn(16, 8))
    tpp = MarkedTPPHead(8, 16, weight, sampled_mark_negatives=4)
    hidden = torch.randn(3, 8)
    delta_t = torch.ones(3)
    target = torch.tensor([1, 2, 3])

    tpp.sampled_mark_nll(hidden, delta_t, target)

    assert tpp._mark_negative_cache.numel() > 0
    assert "_mark_negative_cache" not in tpp.state_dict()


def test_mark_negative_cache_is_reused_and_reset_when_distribution_changes():
    weight = torch.nn.Parameter(torch.randn(16, 8))
    tpp = MarkedTPPHead(8, 16, weight, sampled_mark_negatives=4)

    first = tpp._sample_mark_negatives(4)
    cache_id = tpp._mark_negative_cache.data_ptr()
    offset_after_first = tpp._mark_negative_cache_offset
    second = tpp._sample_mark_negatives(4)

    assert tpp._mark_negative_cache.data_ptr() == cache_id
    assert tpp._mark_negative_cache_offset == offset_after_first + 4
    assert not torch.equal(first, second)

    counts = torch.arange(16, dtype=torch.float32)
    tpp.set_mark_noise_distribution(counts)

    assert tpp._mark_negative_cache.numel() == 0
    assert tpp._mark_negative_cache_offset == 0


def test_mark_loss_samples_negatives_only_while_training(monkeypatch):
    cfg = tiny_config(n_atoms=32)
    model = Genterp(cfg)
    batch = make_batch(n_atoms=cfg.n_atoms)
    out = model(**batch)
    sampled_calls = []

    def sampled_mark_nll(hidden, delta_t, target):
        sampled_calls.append(target.shape[0])
        return hidden.sum() * 0.0

    monkeypatch.setattr(model.tpp, "sampled_mark_nll", sampled_mark_nll)
    model.train()
    marked_tpp_value_loss(
        model.tpp,
        model.value_mod,
        model.value_head,
        model.embed.weight,
        out["hidden"],
        batch["event_ages"],
        batch["target_atoms"],
        batch["event_values"],
        batch["event_pad"],
        batch["censor_age"],
    )

    model.eval()
    marked_tpp_value_loss(
        model.tpp,
        model.value_mod,
        model.value_head,
        model.embed.weight,
        out["hidden"],
        batch["event_ages"],
        batch["target_atoms"],
        batch["event_values"],
        batch["event_pad"],
        batch["censor_age"],
    )

    assert sampled_calls == [int((~batch["event_pad"][:, :-1] & ~batch["event_pad"][:, 1:]).sum().item())]


def test_transcoder_acts():
    cfg = tiny_config()
    model = Genterp(cfg)
    batch = make_batch(n_atoms=cfg.n_atoms)

    out = model(**batch, return_transcoder_acts=True)
    B, T = batch["event_ages"].shape
    assert out["hidden"].shape == (B, T, cfg.dim)
    assert out["pre_mlp"].shape == (B, cfg.n_layers, T, cfg.dim)
    assert out["mlp_out"].shape == (B, cfg.n_layers, T, cfg.dim)
    assert torch.isfinite(out["pre_mlp"]).all()
    assert torch.isfinite(out["mlp_out"]).all()


def test_tpp_sample():
    cfg = tiny_config()
    model = Genterp(cfg).eval()
    batch = make_batch(n_atoms=cfg.n_atoms)
    with torch.no_grad():
        out = model(**batch)
        delta_t, mark = model.tpp.sample(out["hidden"][:, -1])
    assert delta_t.shape == (out["hidden"].shape[0],)
    assert mark.shape == (out["hidden"].shape[0],)
    assert (delta_t > 0).all()
    assert (mark >= 0).all() and (mark < cfg.n_atoms).all()


def test_value_head_sample():
    cfg = tiny_config()
    model = Genterp(cfg).eval()
    _mark_some_atoms_magnitude(model)
    batch = make_batch(n_atoms=cfg.n_atoms)
    with torch.no_grad():
        out = model(**batch)
        leaf = batch["target_atoms"][:, -1].clamp(min=0)
        z = model.value_head.sample(out["hidden"][:, -1], model.embed.weight[leaf])
    assert z.shape == (out["hidden"].shape[0],)
    assert torch.isfinite(z).all()
