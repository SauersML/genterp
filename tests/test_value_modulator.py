from __future__ import annotations

import torch

from genterp.modeling import ValueModulator


def _dense_value_modulation(
    modulator: ValueModulator,
    e_concept: torch.Tensor,
    leaf_atom: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    has_mag = modulator.event_has_magnitude(leaf_atom, value)
    z = modulator.z_score(leaf_atom, value).unsqueeze(-1)
    mlp_input = torch.cat([z, e_concept.float()], dim=-1)
    modulation = torch.tanh(modulator.value_mlp(mlp_input)).to(e_concept.dtype)
    return torch.where(has_mag.unsqueeze(-1), e_concept * modulation, e_concept)


def test_value_modulator_sparse_path_matches_dense_modulation():
    torch.manual_seed(0)
    dim = 16
    n_atoms = 32
    modulator = ValueModulator(dim, n_atoms)
    has_mag = torch.zeros(n_atoms, dtype=torch.bool)
    has_mag[3] = True
    has_mag[9] = True
    modulator.set_stats(torch.linspace(-1.0, 1.0, n_atoms), torch.linspace(0.5, 2.0, n_atoms), has_mag)

    e_concept = torch.randn(2, 4, dim)
    leaf_atom = torch.tensor([[0, 3, 9, 12], [3, 1, 9, 0]])
    value = torch.tensor([[float("nan"), 1.5, -2.0, 7.0], [float("nan"), 1.0, 0.5, -1.0]])

    expected = _dense_value_modulation(modulator, e_concept, leaf_atom, value)
    actual = modulator(e_concept, leaf_atom, value)

    assert torch.allclose(actual, expected)


def test_value_modulator_distinguishes_concepts_at_same_z():
    """z=2 must NOT produce the same modulation for two different concepts
    (e.g., HbA1c vs. sodium); the MLP is conditioned on the concept embedding."""
    torch.manual_seed(0)
    dim = 16
    n_atoms = 8
    modulator = ValueModulator(dim, n_atoms)
    has_mag = torch.ones(n_atoms, dtype=torch.bool)
    modulator.set_stats(torch.zeros(n_atoms), torch.ones(n_atoms), has_mag)
    # Inflate last-layer weight magnitude so distinct concept inputs produce
    # distinct modulation patterns even before any training.
    with torch.no_grad():
        modulator.value_mlp[-1].weight.mul_(50.0)

    e_a = torch.randn(1, 1, dim)
    e_b = torch.randn(1, 1, dim)
    leaf = torch.tensor([[3]])
    value = torch.tensor([[2.0]])  # z = 2 for both atoms (mu=0, sigma=1)

    mod_a = modulator(e_a, leaf, value) / e_a
    mod_b = modulator(e_b, leaf, value) / e_b

    assert not torch.allclose(mod_a, mod_b), "modulation must depend on concept embedding, not only z"


def test_value_modulator_handles_no_selected_values():
    dim = 8
    n_atoms = 16
    modulator = ValueModulator(dim, n_atoms)
    modulator.set_stats(torch.zeros(n_atoms), torch.ones(n_atoms), torch.zeros(n_atoms, dtype=torch.bool))

    e_concept = torch.randn(2, 3, dim)
    leaf_atom = torch.tensor([[1, 2, 3], [4, 5, 6]])
    value = torch.randn(2, 3)

    assert torch.equal(modulator(e_concept, leaf_atom, value), e_concept)
