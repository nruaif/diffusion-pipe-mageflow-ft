"""Visualize StageLR and REX_LR learning rate schedules with matplotlib."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
from torch.optim import AdamW
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from optimizers.stage_lr import StageLR, REX_LR

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lr_plots')
os.makedirs(OUTPUT, exist_ok=True)


def collect_lrs(scheduler, total_steps):
    lrs = [scheduler.optimizer.param_groups[0]['lr']]
    for _ in range(total_steps - 1):
        scheduler.step()
        lrs.append(scheduler.optimizer.param_groups[0]['lr'])
    return lrs


def plot_lr_curve(lrs, title, filename):
    init_lr = lrs[0] if lrs[0] > 0 else 1.0
    normalized = [lr / init_lr for lr in lrs]
    steps = list(range(len(lrs)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax1.plot(steps, lrs, linewidth=1.5, color='#1f77b4')
    ax1.set_ylabel('Learning Rate', fontsize=12)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=init_lr * 1e-3 if init_lr > 0 else 1e-10)

    ax2.plot(steps, normalized, linewidth=1.5, color='#ff7f0e')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('LR (normalized)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.15)

    for ax in (ax1, ax2):
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(9)

    fig.tight_layout()
    path = os.path.join(OUTPUT, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_stage_configs():
    examples = [
        {
            "name": "Warmup -> Linear -> Cosine -> Constant -> Linear to 0",
            "total_iters": 10000,
            "warmup_steps": 100,
            "stages": [
                {"type": "linear", "end_lr": 1e-4, "percent": 0.1},
                {"type": "cosine", "end_lr": 5e-5, "percent": 0.3},
                {"type": "constant", "lr": 5e-5, "percent": 0.3},
                {"type": "linear", "end_lr": 0, "percent": 0.3},
            ],
            "filename": "example1_linear_cosine_constant.png",
        },
        {
            "name": "Warmup -> REX -> Cosine -> Constant",
            "total_iters": 10000,
            "warmup_steps": 50,
            "stages": [
                {"type": "rex", "max_val": 1e-4, "min_val": 1e-5, "percent": 0.3},
                {"type": "cosine", "end_lr": 1e-6, "percent": 0.4},
                {"type": "constant", "lr": 1e-6, "percent": 0.3},
            ],
            "filename": "example2_rex_cosine_constant.png",
        },
        {
            "name": "10% warmup, 30% cosine, 30% REX, 20% constant, 10% linear to 0",
            "total_iters": 10000,
            "warmup_steps": 100,
            "stages": [
                {"type": "linear", "end_lr": 1e-4, "percent": 0.1},
                {"type": "cosine", "end_lr": 5e-5, "percent": 0.3},
                {"type": "rex", "max_val": 1e-4, "min_val": 1e-5, "percent": 0.3},
                {"type": "constant", "lr": 1e-5, "percent": 0.2},
                {"type": "linear", "end_lr": 0, "percent": 0.1},
            ],
            "filename": "example3_full_stage.png",
        },
    ]

    for ex in examples:
        optimizer = AdamW([torch.zeros(1, requires_grad=True)],
                          lr=ex["stages"][0].get("end_lr", ex["stages"][0].get("lr", 1e-4)))
        sched = StageLR(optimizer, stages=ex["stages"], total_iters=ex["total_iters"],
                        warmup_steps=ex["warmup_steps"])
        lrs = collect_lrs(sched, ex["total_iters"] + ex["warmup_steps"])
        print(f"\n  {ex['name']}")
        print(f"    start={lrs[0]:.6e}  end={lrs[-1]:.6e}  min={min(lrs):.6e}  max={max(lrs):.6e}")
        plot_lr_curve(lrs, ex["name"], ex["filename"])


def plot_rex_comparison():
    total = 500
    configs = [
        ("REX 1e-3 -> 1e-5", 1e-3, 1e-5),
        ("REX 1e-4 -> 1e-6", 1e-4, 1e-6),
        ("REX 1e-4 -> 1e-4 (flat)", 1e-4, 1e-4),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, (label, max_val, min_val) in enumerate(configs):
        optimizer = AdamW([torch.zeros(1, requires_grad=True)], lr=max_val)
        sched = REX_LR(optimizer, max_val=max_val, min_val=min_val, num_epochs=total)
        lrs = collect_lrs(sched, total)
        ax.plot(range(total), lrs, linewidth=1.5, label=label, color=colors[idx])
        print(f"  {label}: start={lrs[0]:.6e} end={lrs[-1]:.6e}")

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('REX_LR Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.tight_layout()
    path = os.path.join(OUTPUT, 'rex_comparison.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_warmup_only():
    optimizer = AdamW([torch.zeros(1, requires_grad=True)], lr=1e-3)
    sched = StageLR(optimizer, stages=[{"type": "constant", "lr": 1e-3, "percent": 1.0}],
                    total_iters=1, warmup_steps=100)
    lrs = collect_lrs(sched, 100)
    print(f"\n  Warmup only: start={lrs[0]:.6e} end={lrs[-1]:.6e}")
    plot_lr_curve(lrs, "Warmup only (100 steps, base_lr=1e-3)", "warmup_only.png")


def collect_lrs(scheduler, total_steps):
    lrs = [scheduler.optimizer.param_groups[0]['lr']]
    for _ in range(total_steps - 1):
        scheduler.step()
        lrs.append(scheduler.optimizer.param_groups[0]['lr'])
    return lrs


def plot_lr_curve(lrs, title, filename):
    init_lr = lrs[0] if lrs[0] > 0 else 1.0
    normalized = [lr / init_lr for lr in lrs]
    steps = list(range(len(lrs)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    ax1.plot(steps, lrs, linewidth=1.5, color='#1f77b4')
    ax1.set_ylabel('Learning Rate', fontsize=12)
    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=init_lr * 1e-3 if init_lr > 0 else 1e-10)

    ax2.plot(steps, normalized, linewidth=1.5, color='#ff7f0e')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('LR (normalized)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.15)

    for ax in (ax1, ax2):
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(9)

    fig.tight_layout()
    path = os.path.join(OUTPUT, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    plot_warmup_only()
    plot_rex_comparison()
    plot_stage_configs()
    print("\nDone.")