"""
StageLR — Multi-stage LR scheduler for PyTorch.

Install:
    pip install git+https://github.com/nruaif/diffusion-pipe-mageflow-ft.git#subdirectory=stage_lr
    # or from diffusion-pipe root:
    pip install -e ".[stage-lr]"

Usage:
    from stage_lr import StageLR, REX_LR

    scheduler = StageLR(optimizer,
        stages=[
            {"type": "linear", "end_lr": 1e-4, "percent": 0.1},
            {"type": "cosine", "end_lr": 5e-5, "percent": 0.3},
            {"type": "rex", "max_val": 1e-4, "min_val": 1e-5, "percent": 0.3},
            {"type": "constant", "lr": 1e-5, "percent": 0.3},
        ],
        total_iters=50000,
        warmup_steps=100,
    )
"""

from .scheduler import StageLR, REX_LR

__all__ = ["StageLR", "REX_LR"]
__version__ = "0.1.0"
