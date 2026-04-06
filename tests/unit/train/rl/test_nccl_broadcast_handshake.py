from pathlib import Path

import torch

from prime_rl.inference.vllm.worker.nccl import NCCLWeightUpdateWorker
from prime_rl.trainer.rl.broadcast.nccl import NCCLWeightBroadcast
from prime_rl.utils.nccl import get_nccl_ready_path


class _Logger:
    def debug(self, *_args, **_kwargs) -> None:
        pass


def test_wait_for_nccl_ready_waits_for_each_inference_rank(monkeypatch, tmp_path: Path):
    waited_paths: list[Path] = []

    def fake_wait_for_path(path: Path, interval: float, log_interval: int) -> None:
        waited_paths.append(path)

    monkeypatch.setattr("prime_rl.trainer.rl.broadcast.nccl.sync_wait_for_path", fake_wait_for_path)

    broadcast = NCCLWeightBroadcast.__new__(NCCLWeightBroadcast)
    broadcast.logger = _Logger()
    broadcast.inference_world_size = 2

    broadcast._wait_for_nccl_ready([(0, tmp_path)])

    assert waited_paths == [
        get_nccl_ready_path(tmp_path, 1),
        get_nccl_ready_path(tmp_path, 2),
    ]


def test_update_weights_from_path_signals_ready_before_loading(monkeypatch, tmp_path: Path):
    worker = NCCLWeightUpdateWorker()
    worker.quantize_in_weight_transfer = False
    worker.nccl_broadcast_rank = 2

    class DummyReceiver:
        def receive_state_dict(self):
            yield from ()

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class DummyRunner:
        def __init__(self):
            self.model = DummyModel()
            self.model_config = object()

    ready_path = get_nccl_ready_path(tmp_path, worker.nccl_broadcast_rank)
    observed: dict[str, bool] = {}

    def fake_loader(model, state_iter) -> None:
        observed["ready_exists"] = ready_path.exists()

    def fake_postprocess(model, model_config, device) -> None:
        return None

    worker.nccl_broadcast_receiver = DummyReceiver()
    worker.model_runner = DummyRunner()

    monkeypatch.setattr("prime_rl.inference.vllm.worker.nccl.load_weights_checkpoint", fake_loader)
    monkeypatch.setattr("prime_rl.inference.vllm.worker.nccl.postprocess_weights_checkpoint", fake_postprocess)

    worker.update_weights_from_path(tmp_path.as_posix())

    assert observed["ready_exists"]
    assert ready_path.exists()
