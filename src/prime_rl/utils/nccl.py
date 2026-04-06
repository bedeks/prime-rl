from pathlib import Path

NCCL_READY_MARKER = "NCCL_READY"


def get_nccl_ready_path(weight_dir: Path, rank: int) -> Path:
    """Return the per-rank readiness marker path for NCCL weight transfer."""
    return weight_dir / f"{NCCL_READY_MARKER}.rank{rank}"
