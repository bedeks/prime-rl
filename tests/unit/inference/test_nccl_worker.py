import pytest

from prime_rl.inference.vllm.worker import nccl
from prime_rl.inference.vllm.worker.nccl import resolve_nccl_worker_ranks


@pytest.mark.parametrize(
    (
        "tp_size",
        "tp_rank",
        "dp_rank",
        "dp_world_size",
        "dp_rank_local",
        "rank_offset",
        "gpus_per_server",
        "device_index",
        "local_rank_env",
        "expected",
    ),
    [
        (1, 0, 0, 2, 0, 0, 2, 0, "1", (0, 0, 0, "data_parallel_rank_local")),
        (2, 1, 3, 4, None, 4, 4, 1, "0", (3, 1, 7, "dp_group")),
        (1, 0, 0, 1, None, 0, 2, 0, "1", (1, 1, 1, "local_rank_env")),
        (2, 0, 0, None, None, 0, 4, 2, "2", (2, 1, 2, "local_rank_env")),
    ],
)
def test_resolve_nccl_worker_ranks(
    tp_size: int,
    tp_rank: int,
    dp_rank: int,
    dp_world_size: int | None,
    dp_rank_local: int | None,
    rank_offset: int,
    gpus_per_server: int,
    device_index: int,
    local_rank_env: str,
    expected: tuple[int, int, int, str],
):
    assert (
        resolve_nccl_worker_ranks(
            tp_size=tp_size,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            dp_rank_local=dp_rank_local,
            rank_offset=rank_offset,
            gpus_per_server=gpus_per_server,
            device_index=device_index,
            local_rank_env=local_rank_env,
        )
        == expected
    )


def test_resolve_nccl_worker_ranks_falls_back_to_device_index(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert resolve_nccl_worker_ranks(
        tp_size=2,
        tp_rank=1,
        dp_rank=0,
        dp_world_size=1,
        dp_rank_local=None,
        rank_offset=4,
        gpus_per_server=4,
        device_index=3,
        local_rank_env=None,
    ) == (3, 1, 7, "device_index")


def test_init_broadcaster_preserves_dp_group_path(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nccl, "get_tp_group", lambda: type("Group", (), {"world_size": 2, "rank_in_group": 1})())
    monkeypatch.setattr(nccl, "get_dp_group", lambda: type("Group", (), {"rank_in_group": 3, "world_size": 4})())

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
    monkeypatch.setenv("LOCAL_RANK", "0")

    worker = nccl.NCCLWeightUpdateWorker()
    worker.device = type("Device", (), {"index": 1})()
    worker.parallel_config = type("ParallelConfig", (), {"data_parallel_rank_local": None})()

    worker.init_broadcaster(
        host="localhost",
        port=29501,
        rank_offset=4,
        inference_world_size=8,
        gpus_per_server=4,
        timeout=10,
    )

    assert receiver_init["rank"] == 8
    assert receiver_init["world_size"] == 9


def test_init_broadcaster_prefers_data_parallel_rank_local(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nccl, "get_tp_group", lambda: type("Group", (), {"world_size": 2, "rank_in_group": 1})())
    monkeypatch.setattr(nccl, "get_dp_group", lambda: type("Group", (), {"rank_in_group": 0, "world_size": 1})())

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
    monkeypatch.setenv("LOCAL_RANK", "0")

    worker = nccl.NCCLWeightUpdateWorker()
    worker.device = type("Device", (), {"index": 1})()
    worker.parallel_config = type("ParallelConfig", (), {"data_parallel_rank_local": 1})()

    worker.init_broadcaster(
        host="localhost",
        port=29501,
        rank_offset=4,
        inference_world_size=8,
        gpus_per_server=4,
        timeout=10,
    )

    assert receiver_init["rank"] == 8
    assert receiver_init["world_size"] == 9


def test_init_broadcaster_uses_local_rank_even_when_dp_rank_is_zero(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(nccl, "get_tp_group", lambda: type("Group", (), {"world_size": 1, "rank_in_group": 0})())
    monkeypatch.setattr(nccl, "get_dp_group", lambda: type("Group", (), {"rank_in_group": 0, "world_size": 1})())

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
    worker.parallel_config = type("ParallelConfig", (), {"data_parallel_rank_local": None})()

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
