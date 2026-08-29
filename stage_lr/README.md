# StageLR

Multi-stage LR scheduler for PyTorch — chain `linear`, `cosine`, `constant`, `rex` (REX paper) as percent slices of `total_iters`, with optional linear warmup.

```python
from stage_lr import StageLR

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
```

Install from this repo:

```bash
pip install git+https://github.com/nruaif/diffusion-pipe-mageflow-ft.git#subdirectory=stage_lr
# or from root (includes stage_lr):
pip install -e ".[stage-lr]"  # if root pyproject exposes it
pip install -e ./stage_lr
```

Standalone `REX_LR` also exported: `from stage_lr import REX_LR`.

## Diffusion-pipe TOML

```toml
lr_scheduler = "StageLR"
[StageLR]
warmup_steps = 50
# total_iters optional — defaults to epochs*steps_per_epoch
stages = [
    { type = "linear", end_lr = 1e-4, percent = 0.1 },
    { type = "cosine", end_lr = 5e-5, percent = 0.3 },
    { type = "rex", max_val = 1e-4, min_val = 1e-5, percent = 0.3 },
    { type = "constant", lr = 1e-5, percent = 0.3 },
]
```

Also supports `lr_scheduler = "stage"` / `{type="stage", stages=[...]}` and `[lr_scheduler] type="stage"` for backward compat.
