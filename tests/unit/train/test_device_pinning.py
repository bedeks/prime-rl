from types import SimpleNamespace

from prime_rl.entrypoints.rl import resolve_local_cuda_visibility
from prime_rl.trainer import utils as trainer_utils


def test_resolve_local_cuda_visibility_uses_shared_namespace_for_local_nccl():
    assert resolve_local_cuda_visibility([0, 1], [2], use_shared_nccl_namespace=True) == ("0,1,2", "0,1,2", 2)


def test_resolve_local_cuda_visibility_keeps_role_specific_masks_without_shared_nccl():
    assert resolve_local_cuda_visibility([0, 1], [2], use_shared_nccl_namespace=False) == ("0,1", "2", None)


def test_get_local_cuda_device_id_uses_trainer_base_offset(monkeypatch):
    monkeypatch.setenv("PRIME_TRAINER_CUDA_BASE", "2")
    monkeypatch.setattr(trainer_utils, "get_world", lambda: SimpleNamespace(local_rank=1))

    assert trainer_utils.get_local_cuda_device_id() == 3


def test_get_local_cuda_device_id_defaults_to_local_rank(monkeypatch):
    monkeypatch.delenv("PRIME_TRAINER_CUDA_BASE", raising=False)
    monkeypatch.setattr(trainer_utils, "get_world", lambda: SimpleNamespace(local_rank=1))

    assert trainer_utils.get_local_cuda_device_id() == 1
