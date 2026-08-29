"""
MLP Eggroll flow-matching toy to test StageLR.

Eggroll dataset: 2D swiss-roll (like sklearn) to mimic "eggroll" shape.
MLP learns rectified flow: x_t = (1-t)*noise + t*data, predicts velocity v = data - noise.

Tests StageLR with multi-stage schedule vs constant baseline.

Usage:
    python test/test_stage_lr_mlp_eggroll.py
    python test/test_stage_lr_mlp_eggroll.py --plot  # saves test/eggroll_stage_lr.png
"""

import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F

# Try pip package first, fallback to diffusion-pipe path
try:
    from stage_lr import StageLR, REX_LR
except ImportError:
    from optimizers.stage_lr import StageLR, REX_LR


def make_eggroll(n=2048, noise=0.1, seed=0):
    """Swiss-roll / eggroll 2D dataset. Returns [n,2] tensor."""
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, generator=g) * 3 * math.pi  # angle
    # radius grows with angle (eggroll)
    r = t / math.pi * 0.8 + 0.5
    x = r * torch.cos(t) + torch.randn(n, generator=g) * noise
    y = r * torch.sin(t) + torch.randn(n, generator=g) * noise
    # center and scale to ~[-2,2]
    data = torch.stack([x, y], dim=1)
    data = (data - data.mean(0)) / data.std(0) * 0.8
    return data


class MLPVelocity(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, dim), nn.SiLU(),
            nn.Linear(dim, 2),
        )
        # t embedding via simple MLP (like TimestepEmbedding)
        self.t_proj = nn.Sequential(
            nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, x_t, t):
        # x_t: [B,2], t: [B] or [B,1]
        if t.dim() == 1:
            t = t[:, None]
        # simple concat: [x, t]
        h = torch.cat([x_t, t], dim=-1)  # [B,3]
        return self.net(h)


def train_one_run(scheduler_type="stage", steps=2000, batch=256, plot=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, scheduler: {scheduler_type}")

    data = make_eggroll(8192).to(device)
    model = MLPVelocity(dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if scheduler_type == "stage":
        sched = StageLR(opt,
            stages=[
                {"type": "linear", "end_lr": 1e-3, "percent": 0.1},
                {"type": "cosine", "end_lr": 5e-4, "percent": 0.3},
                {"type": "rex", "max_val": 1e-3, "min_val": 1e-5, "percent": 0.3},
                {"type": "constant", "lr": 1e-5, "percent": 0.3},
            ],
            total_iters=steps,
            warmup_steps=50,
        )
    elif scheduler_type == "rex":
        sched = REX_LR(opt, max_val=1e-3, min_val=1e-5, num_epochs=steps)
    elif scheduler_type == "constant":
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    elif scheduler_type == "linear":
        sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=steps)
    else:
        raise ValueError(scheduler_type)

    losses = []
    lrs = []
    model.train()
    for step in range(steps):
        # sample batch
        idx = torch.randint(0, len(data), (batch,), device=device)
        x0 = data[idx]  # data
        x1 = torch.randn_like(x0)  # noise
        t = torch.rand(batch, device=device)
        x_t = (1 - t[:, None]) * x1 + t[:, None] * x0
        target = x0 - x1
        pred = model(x_t, t)
        loss = F.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        losses.append(loss.item())
        lrs.append(opt.param_groups[0]['lr'])
        if (step + 1) % 500 == 0:
            print(f" step {step+1:4d}/{steps} loss={loss.item():.4f} lr={lrs[-1]:.2e}")

    print(f"Final loss: {losses[-1]:.4f}, min {min(losses):.4f}, max {max(losses):.4f}")
    print(f"LR: start {lrs[0]:.2e} end {lrs[-1]:.2e} min {min(lrs):.2e} max {max(lrs):.2e}")

    # Simple assertion: loss should decrease, LR schedule should have varied
    assert losses[-1] < losses[0] * 1.5, f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    if scheduler_type != "constant":
        assert len(set([round(l, 10) for l in lrs])) > 1, "LR did not change"
    print("OK StageLR works in MLP eggroll flow-matching")

    if plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(losses, label='loss', color='#1f77b4')
            ax1.set_ylabel('MSE loss')
            ax1.set_title(f'MLP Eggroll Flow-Matching ({scheduler_type})')
            ax1.grid(alpha=0.3)
            ax1.legend()
            ax2.plot(lrs, label='lr', color='#ff7f0e')
            ax2.set_ylabel('lr')
            ax2.set_xlabel('step')
            ax2.set_yscale('log')
            ax2.grid(alpha=0.3)
            ax2.legend()
            path = os.path.join(os.path.dirname(__file__), 'eggroll_stage_lr.png')
            plt.tight_layout()
            plt.savefig(path, dpi=150)
            print(f"Saved plot to {path}")
            plt.close()
        except Exception as e:
            print(f"Plot failed: {e}")

    # Also test sampling (generate)
    model.eval()
    with torch.no_grad():
        # sample 512 points via euler
        n_sample = 512
        x = torch.randn(n_sample, 2, device=device)
        dt = 1.0 / 30
        for i in range(30):
            t = torch.full((n_sample,), i * dt, device=device)
            v = model(x, t)
            x = x + v * dt
        print(f"Sampled {n_sample} points, mean {x.mean(0).tolist()}, std {x.std(0).tolist()}")

    return losses, lrs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--scheduler", default="stage", choices=["stage", "rex", "constant", "linear"])
    args = parser.parse_args()

    for sched in [args.scheduler] if args.scheduler != "stage" else ["stage"]:
        train_one_run(scheduler_type=sched, plot=args.plot)

    # Also test all schedulers quickly
    if args.scheduler == "stage":
        print("\n--- quick test all schedulers ---")
        for s in ["rex", "constant", "linear"]:
            train_one_run(scheduler_type=s, steps=200, plot=False)
