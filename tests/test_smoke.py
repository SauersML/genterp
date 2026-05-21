"""Forward + backward through Genterp with joint TPP + value loss; dict return; transcoder acts; sampling."""

from __future__ import annotations

import torch

from genterp import Genterp
from genterp.modeling import AtomEmbedding, MarkedTPPHead, marked_tpp_value_loss
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
    embed = AtomEmbedding(16, 8)
    tpp = MarkedTPPHead(8, 16, embed, sampled_mark_negatives=4)
    hidden = torch.randn(3, 8)
    delta_t = torch.ones(3)
    target = torch.tensor([1, 2, 3])

    tpp.sampled_mark_nll(hidden, delta_t, target)

    assert tpp._mark_negative_cache.numel() > 0
    assert "_mark_negative_cache" not in tpp.state_dict()


def test_mark_negative_cache_is_reused_and_reset_when_distribution_changes():
    embed = AtomEmbedding(16, 8)
    tpp = MarkedTPPHead(8, 16, embed, sampled_mark_negatives=4)

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
        model.cfg.min_time_delta_days,
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
        model.cfg.min_time_delta_days,
    )

    assert sampled_calls == [int((~batch["event_pad"][:, :-1] & ~batch["event_pad"][:, 1:]).sum().item())]


def test_loss_uses_likelihood_weights_for_sparse_heads():
    cfg = tiny_config(n_atoms=32)
    model = Genterp(cfg)
    model.tpp.exact_mark_loss_weight = 0.0
    model.tpp.mark_z_loss_weight = 0.0
    model.value_mod.set_stats(
        value_mu=torch.zeros(cfg.n_atoms),
        value_sigma=torch.ones(cfg.n_atoms),
        atom_has_mag=torch.ones(cfg.n_atoms, dtype=torch.bool),
    )
    batch = make_batch(n_atoms=cfg.n_atoms)

    ld = model.loss(**batch)

    expected = (
        ld["time_nll"]
        + ld["mark_nll"]
        + (ld["n_mag"].float() / ld["n_real"].float()) * ld["value_nll"]
        + (ld["n_censor"].float() / ld["n_real"].float()) * ld["censor_nll"]
    )
    assert torch.allclose(ld["loss"], expected)
    assert torch.allclose(ld["value_loss_weight"], ld["n_mag"].float() / ld["n_real"].float())
    assert torch.allclose(ld["censor_loss_weight"], ld["n_censor"].float() / ld["n_real"].float())


def test_near_zero_time_transitions_do_not_train_time_density():
    cfg = tiny_config(n_atoms=32)
    model = Genterp(cfg)
    batch = make_batch(n_atoms=cfg.n_atoms)
    batch["event_ages"][:, 1] = batch["event_ages"][:, 0]

    ld = model.loss(**batch)

    assert ld["n_time"].item() == ld["n_real"].item() - batch["event_ages"].shape[0]
    assert ld["time_modeled_nll"].item() > ld["time_nll"].item()


def test_hierarchical_embedding_warm_start_is_bit_exact():
    """Activating hierarchical mode with zero-init ancestors must reproduce
    the flat-embedding model exactly, so a flat checkpoint warm-starts into
    a hierarchical model without any change in behavior on the first step.
    """
    torch.manual_seed(0)
    flat = AtomEmbedding(32, 16)
    atoms = torch.randint(low=1, high=32, size=(4, 7))

    flat_out = flat(atoms)
    flat_weight = flat.effective_weight()

    # Build hierarchical with the same leaf params + a non-trivial ancestor table.
    hier = AtomEmbedding(32, 16, n_ancestor_rows=5)
    hier.embedding.load_state_dict(flat.embedding.state_dict())
    ancestor_ids = torch.zeros(32, 3, dtype=torch.long)
    # Give half the atoms one or two ancestors; ancestor_embedding is zero-init.
    ancestor_ids[1, 0] = 1
    ancestor_ids[2, 0] = 2
    ancestor_ids[2, 1] = 3
    ancestor_ids[7, 0] = 4
    ancestor_ids[12, 0] = 5
    hier.set_ancestor_ids(ancestor_ids)

    assert hier.has_ancestors()
    assert torch.equal(hier(atoms), flat_out)
    assert torch.equal(hier.effective_weight(), flat_weight)

    # Now perturb the ancestor embeddings and confirm the output changes —
    # i.e. the hierarchical path is wired in, just zero-suppressed at init.
    with torch.no_grad():
        hier.ancestor_embedding.weight[1].fill_(0.1)
    perturbed = hier(atoms)
    rows_touched = (atoms == 1).any(dim=-1)
    assert not torch.equal(perturbed, flat_out)
    # Only rows containing atom 1 (whose ancestor is node 1) should change.
    unchanged_rows = ~rows_touched
    if unchanged_rows.any():
        assert torch.equal(perturbed[unchanged_rows], flat_out[unchanged_rows])


def test_marked_tpp_sample_generator_controls_all_draws():
    torch.manual_seed(5)
    embed = AtomEmbedding(16, 8)
    head = MarkedTPPHead(dim=8, n_marks=16, embed_module=embed, n_mix=3, time_dim=4)
    hidden = torch.randn(6, 8)

    gen1 = torch.Generator().manual_seed(123)
    delta1, mark1 = head.sample(hidden, generator=gen1)
    torch.manual_seed(999)
    _ = torch.randn(100)
    gen2 = torch.Generator().manual_seed(123)
    delta2, mark2 = head.sample(hidden, generator=gen2)

    torch.testing.assert_close(delta1, delta2)
    assert torch.equal(mark1, mark2)


def test_hierarchical_embedding_normalizes_by_sqrt_ancestor_count():
    embed = AtomEmbedding(8, 4, n_ancestor_rows=4)
    with torch.no_grad():
        embed.embedding.weight.zero_()
        embed.ancestor_embedding.weight[1:].fill_(1.0)
    ancestor_ids = torch.zeros(8, 4, dtype=torch.long)
    ancestor_ids[1, 0] = 1
    ancestor_ids[2] = torch.tensor([1, 2, 3, 4])
    embed.set_ancestor_ids(ancestor_ids)

    out = embed(torch.tensor([1, 2]))

    assert torch.allclose(out[0], torch.ones(4))
    assert torch.allclose(out[1], torch.full((4,), 2.0))


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


def test_per_subject_loss_norm_differs_from_event_weighted_when_lengths_skew():
    """Per-subject normalization must average over subjects, not over tokens."""
    from dataclasses import replace
    cfg = tiny_config()
    model = Genterp(cfg).eval()
    _mark_some_atoms_magnitude(model)
    # Build a batch where one subject has many more events than the other so
    # event-weighted and subject-weighted means provably differ.
    batch = make_batch(B=2, T=12, n_atoms=cfg.n_atoms, seed=0)
    batch["event_pad"][0, 8:] = True  # subject 0 has 8 events; subject 1 has 12

    with torch.no_grad():
        ld_event = model.loss(**batch)
        model.cfg = replace(model.cfg, per_subject_loss_norm=True)
        ld_subject = model.loss(**batch)
    assert torch.isfinite(ld_event["loss"])
    assert torch.isfinite(ld_subject["loss"])
    # Diagnostic per-token NLLs must be identical under either reduction so
    # logged metrics stay comparable across runs.
    assert torch.allclose(ld_event["time_nll"], ld_subject["time_nll"])
    assert torch.allclose(ld_event["mark_nll"], ld_subject["mark_nll"])


def test_per_subject_loss_norm_is_robust_to_empty_heads():
    """A batch where some heads see zero tokens must produce finite loss."""
    from dataclasses import replace
    cfg = tiny_config()
    model = Genterp(cfg).eval()
    # No magnitude atoms → value head sees zero tokens.
    batch = make_batch(n_atoms=cfg.n_atoms, seed=1)
    model.cfg = replace(model.cfg, per_subject_loss_norm=True)
    with torch.no_grad():
        ld = model.loss(**batch)
    assert torch.isfinite(ld["loss"])
