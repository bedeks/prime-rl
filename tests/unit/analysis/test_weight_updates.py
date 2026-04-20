import torch

from prime_rl.analysis.weight_updates import compute_snapshot_update_density


def test_compute_snapshot_update_density_counts_changed_scalars():
    prev_state = {
        "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "bias": torch.tensor([0.0, 1.0]),
    }
    curr_state = {
        "weight": torch.tensor([[1.0, 20.0], [3.0, 40.0]]),
        "bias": torch.tensor([0.0, 1.0]),
    }

    density = compute_snapshot_update_density(prev_state, curr_state, prev_step=0, step=1)

    assert density.prev_step == 0
    assert density.step == 1
    assert density.changed_elements == 2
    assert density.total_elements == 6
    assert density.changed_tensors == 1
    assert density.total_tensors == 2
    assert density.changed_ratio == 2 / 6
    assert density.changed_tensor_ratio == 1 / 2


def test_compute_snapshot_update_density_requires_matching_keys():
    prev_state = {"weight": torch.tensor([1.0])}
    curr_state = {"bias": torch.tensor([1.0])}

    try:
        compute_snapshot_update_density(prev_state, curr_state, prev_step=0, step=1)
    except ValueError as exc:
        assert "State dict keys do not match" in str(exc)
    else:
        raise AssertionError("Expected mismatched keys to raise ValueError")
