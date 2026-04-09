import pytest

from prime_rl.inference.vllm.worker import nccl


def _story(
    *,
    scenario: str,
    current_nccl_rank: int,
    expected_nccl_rank: int,
    rank_source: str,
    tp_size: int,
    tp_rank: int,
    dp_rank: int,
    dp_world_size: int | None,
    dp_rank_local: int | None,
    rank_offset: int,
    device_index: int,
) -> str:
    return (
        f"{scenario}\n"
        f"current implementation: device.index={device_index} -> nccl_rank={current_nccl_rank}\n"
        f"topology-aware reference: rank_source={rank_source} -> nccl_rank={expected_nccl_rank}\n"
        f"inputs: tp_size={tp_size}, tp_rank={tp_rank}, dp_rank={dp_rank}, "
        f"dp_world_size={dp_world_size}, dp_rank_local={dp_rank_local}, rank_offset={rank_offset}"
    )


def resolve_topology_aware_reference(
    *,
    tp_size: int,
    tp_rank: int,
    dp_rank: int,
    dp_world_size: int | None,
    dp_rank_local: int | None,
    rank_offset: int,
    gpus_per_server: int,
    device_index: int,
    local_rank_env: str | None = None,
) -> tuple[int, int, int, str]:
    """Reference rank resolution used to compare against the current local-only implementation."""
    if dp_rank_local is not None:
        local_dp_rank = dp_rank_local
        local_rank = local_dp_rank * tp_size + tp_rank
        global_rank_inference = rank_offset + local_rank
        return local_rank, local_dp_rank, global_rank_inference, "data_parallel_rank_local"

    if dp_world_size is not None and dp_world_size > 1:
        local_dp_rank = dp_rank % (gpus_per_server // tp_size)
        local_rank = local_dp_rank * tp_size + tp_rank
        global_rank_inference = rank_offset + local_rank
        return local_rank, local_dp_rank, global_rank_inference, "dp_group"

    local_rank = int(local_rank_env) if local_rank_env is not None else device_index
    local_dp_rank = local_rank // tp_size
    global_rank_inference = rank_offset + local_rank
    return local_rank, local_dp_rank, global_rank_inference, "local_rank_env"


def _capture_receiver_init(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    receiver_init: dict[str, object] = {}

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
    return receiver_init


def test_init_broadcaster_uses_device_index_for_rank(monkeypatch: pytest.MonkeyPatch):
    receiver_init = _capture_receiver_init(monkeypatch)

    worker = nccl.NCCLWeightUpdateWorker()
    worker.device = type("Device", (), {"index": 1})()

    worker.init_broadcaster(
        host="localhost",
        port=29501,
        rank_offset=4,
        inference_world_size=8,
        timeout=10,
    )

    assert receiver_init["rank"] == 6
    assert receiver_init["world_size"] == 9


def test_topology_aware_reference_prefers_data_parallel_rank_local():
    assert (
        resolve_topology_aware_reference(
            tp_size=2,
            tp_rank=1,
            dp_rank=0,
            dp_world_size=1,
            dp_rank_local=1,
            rank_offset=4,
            gpus_per_server=4,
            device_index=1,
            local_rank_env="0",
        )
        == (3, 1, 7, "data_parallel_rank_local")
    )


def test_rank_story_data_parallel_rank_local_case():
    _, _, expected_global_rank, rank_source = resolve_topology_aware_reference(
        tp_size=2,
        tp_rank=1,
        dp_rank=0,
        dp_world_size=1,
        dp_rank_local=1,
        rank_offset=4,
        gpus_per_server=4,
        device_index=1,
        local_rank_env="0",
    )
    current_nccl_rank = 6
    expected_nccl_rank = expected_global_rank + 1

    story = _story(
        scenario="current local-only rank resolution ignores data_parallel_rank_local",
        current_nccl_rank=current_nccl_rank,
        expected_nccl_rank=expected_nccl_rank,
        rank_source=rank_source,
        tp_size=2,
        tp_rank=1,
        dp_rank=0,
        dp_world_size=1,
        dp_rank_local=1,
        rank_offset=4,
        device_index=1,
    )

    assert current_nccl_rank == 6
    assert expected_nccl_rank == 8
    assert "current implementation: device.index=1 -> nccl_rank=6" in story
    assert "topology-aware reference: rank_source=data_parallel_rank_local -> nccl_rank=8" in story


def test_rank_story_dp_group_case():
    _, _, expected_global_rank, rank_source = resolve_topology_aware_reference(
        tp_size=2,
        tp_rank=1,
        dp_rank=3,
        dp_world_size=4,
        dp_rank_local=None,
        rank_offset=4,
        gpus_per_server=4,
        device_index=1,
        local_rank_env="0",
    )
    current_nccl_rank = 6
    expected_nccl_rank = expected_global_rank + 1

    story = _story(
        scenario="current local-only rank resolution ignores TP-aware flattening of a real DP group",
        current_nccl_rank=current_nccl_rank,
        expected_nccl_rank=expected_nccl_rank,
        rank_source=rank_source,
        tp_size=2,
        tp_rank=1,
        dp_rank=3,
        dp_world_size=4,
        dp_rank_local=None,
        rank_offset=4,
        device_index=1,
    )

    assert current_nccl_rank == 6
    assert expected_nccl_rank == 8
    assert "current implementation: device.index=1 -> nccl_rank=6" in story
    assert "topology-aware reference: rank_source=dp_group -> nccl_rank=8" in story


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Current implementation uses self.device.index only. "
        "For tp=2 with data_parallel_rank_local=1, the topology-aware NCCL rank is 8, not 6."
    ),
)
def test_current_impl_misranks_when_data_parallel_rank_local_is_available(monkeypatch: pytest.MonkeyPatch):
    receiver_init = _capture_receiver_init(monkeypatch)

    worker = nccl.NCCLWeightUpdateWorker()
    worker.device = type("Device", (), {"index": 1})()

    worker.init_broadcaster(
        host="localhost",
        port=29501,
        rank_offset=4,
        inference_world_size=8,
        timeout=10,
    )

    _, _, expected_global_rank, rank_source = resolve_topology_aware_reference(
        tp_size=2,
        tp_rank=1,
        dp_rank=0,
        dp_world_size=1,
        dp_rank_local=1,
        rank_offset=4,
        gpus_per_server=4,
        device_index=1,
        local_rank_env="0",
    )
    expected_nccl_rank = expected_global_rank + 1

    assert receiver_init["rank"] == expected_nccl_rank, _story(
        scenario="current local-only rank resolution ignores data_parallel_rank_local",
        current_nccl_rank=int(receiver_init["rank"]),
        expected_nccl_rank=expected_nccl_rank,
        rank_source=rank_source,
        tp_size=2,
        tp_rank=1,
        dp_rank=0,
        dp_world_size=1,
        dp_rank_local=1,
        rank_offset=4,
        device_index=1,
    )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Current implementation ignores TP-aware flattening of a real multi-rank DP group. "
        "For tp=2 and dp_rank=3/world_size=4, the topology-aware NCCL rank is 8, not 6."
    ),
)
def test_current_impl_misranks_when_dp_group_is_meaningful(monkeypatch: pytest.MonkeyPatch):
    receiver_init = _capture_receiver_init(monkeypatch)

    worker = nccl.NCCLWeightUpdateWorker()
    worker.device = type("Device", (), {"index": 1})()

    worker.init_broadcaster(
        host="localhost",
        port=29501,
        rank_offset=4,
        inference_world_size=8,
        timeout=10,
    )

    _, _, expected_global_rank, rank_source = resolve_topology_aware_reference(
        tp_size=2,
        tp_rank=1,
        dp_rank=3,
        dp_world_size=4,
        dp_rank_local=None,
        rank_offset=4,
        gpus_per_server=4,
        device_index=1,
        local_rank_env="0",
    )
    expected_nccl_rank = expected_global_rank + 1

    assert receiver_init["rank"] == expected_nccl_rank, _story(
        scenario="current local-only rank resolution ignores TP-aware flattening of a real DP group",
        current_nccl_rank=int(receiver_init["rank"]),
        expected_nccl_rank=expected_nccl_rank,
        rank_source=rank_source,
        tp_size=2,
        tp_rank=1,
        dp_rank=3,
        dp_world_size=4,
        dp_rank_local=None,
        rank_offset=4,
        device_index=1,
    )
