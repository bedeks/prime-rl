import pytest

from prime_rl.inference.vllm.worker import nccl
from prime_rl.inference.vllm.worker.nccl import resolve_nccl_worker_ranks


@pytest.mark.parametrize(
    ("tp_size", "rank_offset", "device_index", "local_rank_env", "expected"),
    [
        (1, 0, 0, "0", (0, 0, 0)),
        (1, 0, 1, "1", (1, 1, 1)),
        (2, 0, 0, "0", (0, 0, 0)),
        (2, 0, 1, "1", (1, 0, 1)),
        (2, 0, 2, "2", (2, 1, 2)),
        (2, 4, 1, "1", (1, 0, 5)),
    ],
)
def test_resolve_nccl_worker_ranks_prefers_local_rank(
    tp_size: int,
    rank_offset: int,
    device_index: int,
    local_rank_env: str,
    expected: tuple[int, int, int],
):
    assert (
        resolve_nccl_worker_ranks(
            tp_size=tp_size,
            rank_offset=rank_offset,
            device_index=device_index,
            local_rank_env=local_rank_env,
        )
        == expected
    )


def test_resolve_nccl_worker_ranks_falls_back_to_device_index():
    assert resolve_nccl_worker_ranks(tp_size=2, rank_offset=4, device_index=3, local_rank_env=None) == (3, 1, 7)


def test_init_broadcaster_uses_local_rank_even_when_dp_rank_is_zero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nccl, "get_tp_group", lambda: type("Group", (), {"world_size": 1, "rank_in_group": 0})())
    monkeypatch.setattr(nccl, "get_dp_group", lambda: type("Group", (), {"rank_in_group": 0})())

    receiver_init = {}

    class FakeReceiver:
        def __init__(self, host, port, rank, world_size, device, timeout):
            receiver_init.update(
                {
                    "host": host,
                    "port": port,
                    "rank": rank,
                    "world_size": world_size,
                    "device": device,
                    "timeout": timeout,
                }
            )

    monkeypatch.setattr(nccl, "NCCLWeightBroadcastReceiver", FakeReceiver)
    monkeypatch.setenv("LOCAL_RANK", "1")

    worker = nccl.NCCLWeightUpdateWorker()
    worker.device = type("Device", (), {"index": 1})()

    worker.init_broadcaster(
        host="localhost",
        port=29501,
        rank_offset=0,
        inference_world_size=2,
        gpus_per_server=2,
        timeout=10,
    )

    assert receiver_init["rank"] == 2
    assert receiver_init["world_size"] == 3
