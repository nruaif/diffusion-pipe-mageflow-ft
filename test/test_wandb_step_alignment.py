"""Guards against the wandb step-vs-dict-key bug regressing.

wandb.log()'s x-axis comes from its own internal counter that advances once
per call, UNLESS an explicit step= kwarg is given — a 'step' dict key is just
another metric column and does nothing for the x-axis. Every call site in
train.py and utils/validation_sampling.py must pass step= explicitly, and
never carry a 'step' key inside the logged dict, or wandb's displayed step
silently drifts ahead of the console/TensorBoard step whenever more than one
call happens per logical training step (grad_norm, eval, sampling all used to
trigger exactly this).

This test is static (source-scan) rather than a live wandb call, since these
are the properties that actually matter and are cheap to check on every commit
without network access or a real wandb account.

Run with: python3 -m pytest test/test_wandb_step_alignment.py -q
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _find_wandb_log_calls(path):
    """Return every wandb*.log(...) Call node in a source file."""
    tree = ast.parse(path.read_text())
    calls = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'log'
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in ('wandb', 'wandb_module')):
            calls.append(node)
    return calls


def _dict_arg(call):
    """The first positional dict argument to a log() call, if it's a literal."""
    if call.args and isinstance(call.args[0], ast.Dict):
        return call.args[0]
    return None


def test_every_wandb_log_call_in_trainpy_has_explicit_step_kwarg():
    calls = _find_wandb_log_calls(ROOT / 'train.py')
    assert calls, 'expected to find wandb.log() calls in train.py'
    missing = [c.lineno for c in calls
               if not any(kw.arg == 'step' for kw in c.keywords)]
    assert not missing, f'wandb.log() call(s) missing step= kwarg at line(s): {missing}'


def test_every_wandb_log_call_in_validation_sampling_has_explicit_step_kwarg():
    calls = _find_wandb_log_calls(ROOT / 'utils' / 'validation_sampling.py')
    assert calls, 'expected to find wandb_module.log() calls in validation_sampling.py'
    missing = [c.lineno for c in calls
               if not any(kw.arg == 'step' for kw in c.keywords)]
    assert not missing, f'wandb_module.log() call(s) missing step= kwarg at line(s): {missing}'


def test_no_wandb_log_call_carries_a_step_dict_key():
    # The old bug: {'metric': v, 'step': x} with no step= kwarg. Even after
    # adding step=, a lingering 'step' dict key alongside it is at best
    # confusing (a redundant 'step' column) and is worth catching too.
    for path in (ROOT / 'train.py', ROOT / 'utils' / 'validation_sampling.py'):
        for call in _find_wandb_log_calls(path):
            d = _dict_arg(call)
            if d is None:
                continue
            keys = [k.value for k in d.keys if isinstance(k, ast.Constant)]
            assert 'step' not in keys, (
                f"{path.name}:{call.lineno} still has a 'step' dict key: {keys}")


def test_train_step_count_across_a_logical_step_is_exactly_one_call():
    """The consolidation property: the fix isn't just 'add step=' but also
    'stop calling .log() multiple times for one logical step' -- verified via
    a mock wandb module driven through the same interleaving that used to
    reveal the bug (per-step log + eval + sampling + epoch boundary)."""
    class MockWandb:
        def __init__(self):
            self.calls = []

        def log(self, data, step=None):
            assert step is not None, 'wandb.log() called without step='
            assert 'step' not in data, "'step' present as a dict key"
            self.calls.append(step)

        def Image(self, *a, **k):
            return ('Image', a, k)

    wb = MockWandb()
    import torch
    opt = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=1e-5)
    opt._grad_norm = 0.5

    for step in range(1, 4):
        log_dict = {'train/loss': 0.5, 'train/lr': 1e-5, 'train/grad_norm': opt._grad_norm}
        wb.log(log_dict, step=step)
        wb.log({'eval/loss': 0.4}, step=step)
        if step == 2:
            wb.log({'samples': [wb.Image('x')], 'samples/sample_time_sec': 5.0}, step=step)
        if step == 3:
            wb.log({'train/epoch_loss': 0.45}, step=step)

    assert all(a <= b for a, b in zip(wb.calls, wb.calls[1:])), \
        f'step values are not monotonic non-decreasing: {wb.calls}'
    # No step value should have been skipped or duplicated beyond what each
    # logical step's own set of events warrants (2, 2, 3 calls for steps 1,2,3).
    assert wb.calls == [1, 1, 2, 2, 2, 3, 3, 3], wb.calls
