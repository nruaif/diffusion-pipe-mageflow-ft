import math
from torch.optim.lr_scheduler import _LRScheduler


class REX_LR(_LRScheduler):
    """REX learning rate scheduler from 'REX: Revisiting Budgeted Training with an Improved Schedule'.

    Args:
        optimizer: PyTorch optimizer.
        max_val: Maximum learning rate (at the start of the schedule).
        min_val: Minimum learning rate (at the end of the schedule).
        num_epochs: Number of steps in the schedule.
        last_epoch: Index of the last epoch.
    """

    def __init__(self, optimizer, max_val, min_val, num_epochs=1, last_epoch=-1):
        self.num_epochs = num_epochs
        self.min_val = min_val
        self.max_val = max_val
        if not self.min_val <= self.max_val:
            raise ValueError(f'min_val ({min_val}) must be <= max_val ({max_val})')
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        mod_iter = float(self.last_epoch % self.num_epochs)
        z = float(self.num_epochs - mod_iter) / self.num_epochs
        val = self.min_val + float(self.max_val - self.min_val) * (z / (1 - 0.9 + 0.9 * z))
        return [val]


class StageLR(_LRScheduler):
    """Multi-stage learning rate scheduler.

    Chains together multiple schedule stages (linear, cosine, constant, rex)
    defined as fractions of total training iterations. Optionally precedes
    them with a linear warmup phase.

    Config example (TOML):

        [StageLR]
        warmup_steps = 100
        total_iters = 50000  # optional, otherwise epochs*steps_per_epoch
        stages = [
            { type = "linear",   end_lr = 1e-4, percent = 0.1 },
            { type = "cosine",   end_lr = 5e-5, percent = 0.3 },
            { type = "rex",      max_val = 1e-4, min_val = 1e-5, percent = 0.3 },
            { type = "constant", lr = 1e-5,      percent = 0.3 },
        ]

    Or inline dict:

        lr_scheduler = { type = "stage", warmup_steps = 100, stages = [...] }
    """

    def __init__(self, optimizer, stages, total_iters, warmup_steps=0, last_epoch=-1):
        if not stages:
            raise ValueError('stages list must not be empty')
        if total_iters <= 0:
            raise ValueError(f'total_iters must be > 0, got {total_iters}')

        self.total_iters = total_iters
        self.warmup_steps = warmup_steps
        self.stage_info = self._build_stage_info(stages, total_iters)
        self._initial_lr = [group['lr'] for group in optimizer.param_groups]
        super().__init__(optimizer, last_epoch)

    def _build_stage_info(self, stages, total_iters):
        boundaries = []
        cumulative = 0
        for s in stages:
            n_steps = max(1, round(s['percent'] * total_iters))
            if cumulative + n_steps > total_iters:
                n_steps = total_iters - cumulative
            if n_steps <= 0:
                continue
            stype = s['type']
            if stype not in ('linear', 'cosine', 'constant', 'rex'):
                raise ValueError(f'Unknown stage type: {stype}')
            if stype in ('linear', 'cosine') and 'end_lr' not in s:
                raise ValueError(f'Stage type "{stype}" requires "end_lr"')
            if stype == 'constant' and 'lr' not in s:
                raise ValueError('Stage type "constant" requires "lr"')
            if stype == 'rex' and ('max_val' not in s or 'min_val' not in s):
                raise ValueError('Stage type "rex" requires "max_val" and "min_val"')
            boundaries.append({
                'type': stype,
                'params': s,
                'start': cumulative,
                'end': cumulative + n_steps,
                'n_steps': n_steps,
            })
            cumulative += n_steps
        return boundaries

    def get_lr(self):
        step = self.last_epoch

        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self._warmup_lr(step)

        step -= self.warmup_steps
        step = max(step, 0)
        if step >= self.total_iters:
            step = self.total_iters - 1

        for idx, stage in enumerate(self.stage_info):
            if stage['start'] <= step < stage['end']:
                start_lr = self._get_stage_start_lr(idx)
                return self._compute_stage_lr(stage, step, start_lr)

        last_stage = self.stage_info[-1]
        start_lr = self._get_stage_start_lr(len(self.stage_info) - 1)
        return self._compute_stage_lr(last_stage, last_stage['end'] - 1, start_lr)

    def _warmup_lr(self, step):
        if self.warmup_steps <= 1:
            return [group['lr'] for group in self.optimizer.param_groups]
        factor = (1.0 / self.warmup_steps + (1.0 - 1.0 / self.warmup_steps) * step / self.warmup_steps)
        return [base_lr * factor for base_lr in self._initial_lr]

    def _get_stage_start_lr(self, idx):
        if idx == 0:
            return self._initial_lr[0]
        prev_stage = self.stage_info[idx - 1]
        last_step = prev_stage['end'] - 1
        prev_start_lr = self._get_stage_start_lr(idx - 1)
        return self._compute_stage_lr(prev_stage, last_step, prev_start_lr)[0]

    def _compute_stage_lr(self, stage, step, start_lr):
        stage_type = stage['type']
        params = stage['params']
        n_steps = stage['n_steps']
        local_step = step - stage['start']
        progress = local_step / max(n_steps - 1, 1)

        if stage_type == 'linear':
            end_lr = params['end_lr']
            return [start_lr + (end_lr - start_lr) * progress]
        elif stage_type == 'cosine':
            end_lr = params['end_lr']
            return [start_lr + (end_lr - start_lr) * (1 - math.cos(math.pi * progress)) / 2]
        elif stage_type == 'constant':
            return [params['lr']]
        elif stage_type == 'rex':
            max_val = params['max_val']
            min_val = params['min_val']
            z = float(n_steps - local_step) / n_steps
            val = min_val + (max_val - min_val) * (z / (1 - 0.9 + 0.9 * z))
            return [val]
        else:
            raise NotImplementedError(f'Unknown stage type: {stage_type}')
