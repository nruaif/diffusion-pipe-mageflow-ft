"""Tests for LR logging helpers in train.py.

These cover the extraction path only (no DeepSpeed, no GPU): that LR is read
correctly for every optimizer/scheduler shape this repo can construct, and that
a malformed or exotic optimizer degrades logging instead of crashing a run.

Run with: python3 -m pytest test/test_lr_logging.py -q
"""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))




# train.py guards everything behind `if __name__ == '__main__':`, so importing
# it is safe, but it also imports deepspeed at module scope. Extract just the
# helper functions instead.
def _extract(names):
    import ast
    src = (ROOT / 'train.py').read_text()
    tree = ast.parse(src)
    wanted = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(wanted) == len(names), f'missing helpers: {names}'
    mod = ast.Module(body=wanted, type_ignores=[])
    ns = {}
    exec(compile(mod, '<train_helpers>', 'exec'), ns)
    return ns


_ns = _extract({'_get_param_group_lrs', '_format_lrs'})
_get_param_group_lrs = _ns['_get_param_group_lrs']
_format_lrs = _ns['_format_lrs']


def make_optimizer(lr=0.1, n_groups=1, lrs=None):
    params = [torch.nn.Parameter(torch.ones(2, 2)) for _ in range(max(n_groups, 1))]
    if lrs is not None:
        groups = [{'params': [p], 'lr': l} for p, l in zip(params, lrs)]
        return torch.optim.SGD(groups, lr=lr)
    return torch.optim.SGD([{'params': [p]} for p in params], lr=lr)


# --- basic extraction ------------------------------------------------------

def test_reads_lr_from_a_plain_optimizer():
    assert _get_param_group_lrs(make_optimizer(lr=1e-5)) == [1e-5]


def test_identical_groups_are_deduplicated():
    opt = make_optimizer(lr=3e-4, n_groups=4)
    assert _get_param_group_lrs(opt) == [3e-4]


def test_differing_groups_are_all_reported_in_order():
    opt = make_optimizer(n_groups=3, lrs=[1e-4, 5e-5, 1e-5])
    assert _get_param_group_lrs(opt) == [1e-4, 5e-5, 1e-5]


# --- degradation instead of crashing --------------------------------------

def test_optimizer_without_param_groups_returns_empty():
    class NoGroups:
        pass
    assert _get_param_group_lrs(NoGroups()) == []


def test_param_group_missing_lr_is_skipped_not_fatal():
    class FakeOpt:
        param_groups = [{'params': []}, {'params': [], 'lr': 2e-5}]
    assert _get_param_group_lrs(FakeOpt()) == [2e-5]


def test_all_groups_missing_lr_returns_empty():
    class FakeOpt:
        param_groups = [{'params': []}]
    assert _get_param_group_lrs(FakeOpt()) == []


def test_non_dict_param_group_is_skipped():
    class FakeOpt:
        param_groups = ['not a dict', {'lr': 1e-4}]
    assert _get_param_group_lrs(FakeOpt()) == [1e-4]


def test_tensor_lr_is_coerced_to_float():
    class FakeOpt:
        param_groups = [{'lr': torch.tensor(1e-4)}]
    out = _get_param_group_lrs(FakeOpt())
    assert out == [pytest.approx(1e-4)] and isinstance(out[0], float)


# --- formatting ------------------------------------------------------------

def test_format_single_lr():
    assert _format_lrs([1e-5]) == '1.000e-05'


def test_format_multiple_lrs():
    assert _format_lrs([1e-4, 1e-5]) == '1.000e-04 / 1.000e-05'


def test_format_empty_is_not_a_crash():
    assert _format_lrs([]) == 'n/a'


# --- integration with the schedulers this repo actually builds -------------

def test_tracks_constant_lr_scheduler():
    opt = make_optimizer(lr=1e-5)
    sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    for _ in range(5):
        opt.step()
        sched.step()
    assert _get_param_group_lrs(opt) == [pytest.approx(1e-5)]


def test_tracks_sequentiallr_warmup_path():
    # The path where scheduler.get_last_lr() has historically raised.
    opt = make_optimizer(lr=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=5)
    main = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warmup, main], milestones=[5])

    seen = []
    for _ in range(10):
        opt.step()
        sched.step()
        seen.append(_get_param_group_lrs(opt)[0])

    assert seen[0] < seen[4], 'LR should rise during warmup'
    assert seen[-1] == pytest.approx(1e-4), 'LR should reach the configured value'


def test_tracks_linear_decay_to_zero():
    opt = make_optimizer(lr=1e-4)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.0, total_iters=10)
    for _ in range(10):
        opt.step()
        sched.step()
    assert _get_param_group_lrs(opt)[0] == pytest.approx(0.0, abs=1e-9)


def test_lr_changes_are_actually_observable_over_a_schedule():
    # Guards against the logging silently reporting a constant when it isn't.
    opt = make_optimizer(lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    seen = set()
    for _ in range(20):
        opt.step()
        sched.step()
        seen.add(round(_get_param_group_lrs(opt)[0], 12))
    assert len(seen) > 5, 'expected a varying LR to produce varying log values'
