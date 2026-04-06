import os
import pickle
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Generator, cast

import torch
from torch.nn import Module
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger

from prime_rl.inference.vllm.worker.weight_transfer import (
    load_weights_checkpoint,
    load_weights_kernel,
    postprocess_weights_checkpoint,
    postprocess_weights_kernel,
)
from prime_rl.utils.nccl import get_nccl_ready_path

# This is to get type hints for the Worker class but not actually extend it at runtime as this is required by vLLM worker extension
if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

    Worker = Worker
else:
    Worker = object

logger = init_logger("vllm.inference.vllm.worker_nccl")


def resolve_nccl_worker_ranks(
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
    """Resolve per-worker NCCL ranks for single-server inference shards.

    Prefer vLLM's explicit local DP rank when available. Otherwise preserve the
    old DP-group based behavior when the worker is part of a real multi-rank DP
    group. Only fall back to the worker's local GPU rank in the singleton-group
    layout where DP rank cannot distinguish local shards.
    """
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

    local_rank_value = local_rank_env if local_rank_env is not None else os.environ.get("LOCAL_RANK")
    rank_source = "local_rank_env" if local_rank_value is not None else "device_index"
    local_rank = int(local_rank_value) if local_rank_value is not None else device_index
    local_dp_rank = local_rank // tp_size
    global_rank_inference = rank_offset + local_rank
    return local_rank, local_dp_rank, global_rank_inference, rank_source


def receive_integer(communicator: PyNcclCommunicator) -> int:
    """Receive an integer from the trainer master rank using NCCL communicator."""
    integer_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(integer_tensor, src=0)
    return cast(int, integer_tensor.item())


def receive_state_dict(communicator: PyNcclCommunicator) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Stream tensors in a state dict broadcasted over NCCL."""
    size_tensor = torch.tensor([10], dtype=torch.long).to(communicator.device)
    communicator.broadcast(size_tensor, src=0)
    state_tensor = torch.empty(cast(int, size_tensor.item()), dtype=torch.uint8).to(communicator.device)
    communicator.broadcast(state_tensor, src=0)

    metadata = pickle.loads(bytes(state_tensor.cpu().numpy()))

    # Receive concatenated tensors per dtype and split them back
    for dtype, tensor_info_list in metadata.items():
        # Receive concatenated tensor for this dtype
        total_elements = sum(numel for _, _, numel in tensor_info_list)
        concatenated = torch.empty(total_elements, dtype=dtype, device=communicator.device)
        communicator.broadcast(concatenated, src=0)

        # Split concatenated tensor back into individual tensors
        offset = 0
        for key, shape, numel in tensor_info_list:
            tensor = concatenated[offset : offset + numel].view(shape).clone()
            offset += numel
            try:
                yield key, tensor
            finally:
                del tensor

        del concatenated


class NCCLWeightBroadcastReceiver:
    def __init__(
        self,
        host: str,
        port: int,
        rank: int,
        world_size: int,
        device: int | str | torch.device,
        timeout: int,
    ):
        logger.info(f"Initializing NCCL broadcast receiver ({host}:{port}, rank={rank}, world_size={world_size})")

        pg = StatelessProcessGroup.create(host=host, port=port, rank=rank, world_size=world_size, store_timeout=timeout)
        self.communicator = PyNcclCommunicator(pg, device=device)

    @torch.no_grad()
    def receive_state_dict(self):
        """Receives the state dict of a model from the trainer master rank using NCCL communicator."""
        logger.info("Receiving weights from trainer")
        num_state_dict_to_receive = receive_integer(self.communicator)
        logger.info(f"Receiving {num_state_dict_to_receive} layer state dicts")
        for layer_id in range(num_state_dict_to_receive):
            logger.info(f"Receiving state dict {layer_id + 1}/{num_state_dict_to_receive}")
            for key, value in receive_state_dict(self.communicator):
                yield key, value


class NCCLWeightUpdateWorker(Worker):
    """vLLM worker extension for updating weights in-place using NCCL."""

    def init_broadcaster(
        self,
        host: str,
        port: int,
        rank_offset: int,
        inference_world_size: int,
        gpus_per_server: int,
        timeout: int,
        quantize_in_weight_transfer: bool = False,
    ) -> None:
        """Initialize the NCCL broadcast receiver.

        Args:
            rank_offset: Starting GPU offset for this server in the global inference group.
            inference_world_size: Total number of inference GPUs across all servers.
            gpus_per_server: Number of GPUs managed by this server instance.
        """
        self.quantize_in_weight_transfer = quantize_in_weight_transfer
        tp_group = get_tp_group()
        dp_group = get_dp_group()
        tp_size = tp_group.world_size
        tp_rank = tp_group.rank_in_group
        dp_rank = dp_group.rank_in_group
        dp_world_size = getattr(dp_group, "world_size", None)
        parallel_config = getattr(self, "parallel_config", None)
        dp_rank_local = getattr(parallel_config, "data_parallel_rank_local", None)

        local_rank, local_dp_rank, global_rank_inference, rank_source = resolve_nccl_worker_ranks(
            tp_size=tp_size,
            tp_rank=tp_rank,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            dp_rank_local=dp_rank_local,
            rank_offset=rank_offset,
            gpus_per_server=gpus_per_server,
            device_index=self.device.index,
        )

        logger.info(
            f"Worker [tp={tp_rank} dp={dp_rank} dp_world_size={dp_world_size} "
            f"dp_rank_local={dp_rank_local} local_dp={local_dp_rank} "
            f"rank_source={rank_source} rank_offset={rank_offset}] "
            f"-> [global_rank={global_rank_inference} inference_world_size={inference_world_size}]"
        )

        self.nccl_broadcast_receiver = NCCLWeightBroadcastReceiver(
            host=host,
            port=port,
            rank=global_rank_inference + 1,  # +1 as the trainer broadcaster is on rank 0
            world_size=inference_world_size + 1,  # +1 as the trainer broadcaster is on rank 0
            device=self.device,
            timeout=timeout,
        )
        self.nccl_broadcast_rank = global_rank_inference + 1

    def update_weights_from_path(self, weight_dir: str) -> None:
        """Update weights with the nccl communicator."""
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        state_iter = self.nccl_broadcast_receiver.receive_state_dict()
        device = next(model.parameters()).device
        loader_fn: Callable[[Module, Generator[tuple[str, torch.Tensor], None, None]], None]
        postprocess_fn: Callable[[Module, object, torch.device], None]
        if self.quantize_in_weight_transfer:
            loader_fn = load_weights_kernel
            postprocess_fn = postprocess_weights_kernel
        else:
            loader_fn = load_weights_checkpoint
            postprocess_fn = postprocess_weights_checkpoint

        ready_path = get_nccl_ready_path(Path(weight_dir), self.nccl_broadcast_rank)
        ready_path.parent.mkdir(parents=True, exist_ok=True)
        ready_path.touch()
        logger.info(f"Signaled NCCL readiness at {ready_path}")
        loader_fn(model, state_iter)
        postprocess_fn(model, self.model_runner.model_config, device)
