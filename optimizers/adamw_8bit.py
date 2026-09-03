import warnings

import torch
import bitsandbytes
import bitsandbytes.functional as F


# Keys that older bitsandbytes exposed via get_config()/__init__ but which
# modern versions (>= ~0.45) removed entirely:
#   - percentile_clipping (and its state["gnorm_vec"] buffer)
#   - block_wise          (the non-blockwise 8-bit path was dropped)
# Passing them to the constructor now raises TypeError, and reading them out
# of get_config() now raises KeyError, so this optimizer tolerates both the
# old and new bitsandbytes rather than pinning to one.
_LEGACY_BNB_KWARGS = ('percentile_clipping', 'block_wise')


class AdamW8bitKahan(bitsandbytes.optim.AdamW8bit):
    """AdamW8bit with Kahan summation compensation.

    The 'shift' buffer is the Kahan compensation term: bitsandbytes writes the
    parameter update into `shift` instead of directly into `p`, then the tail
    of update_step folds it into `p` while carrying the lost low-order bits
    forward. That recovers most of the precision lost by keeping master weights
    in bf16, which is what makes bf16 training viable without an fp32 copy.

    Because Kahan compensation is intrinsic to this class, there is no
    `kahan_sum` toggle -- it is always on. A `kahan_sum` kwarg is accepted and
    ignored (with a warning if someone tries to disable it) so that configs
    written for optimi-style optimizers don't hard-crash here.
    """

    def __init__(self, *args, stabilize=False, kahan_sum=None, **kwargs):
        if kahan_sum is False:
            warnings.warn(
                'AdamW8bitKahan: kahan_sum=false was requested but Kahan summation is '
                'intrinsic to this optimizer and cannot be disabled. Use type = '
                "'adamw8bit' if you want plain AdamW8bit without compensation.")
        # Drop legacy kwargs modern bitsandbytes no longer accepts, rather than
        # letting them reach __init__ as a TypeError.
        for key in _LEGACY_BNB_KWARGS:
            if key in kwargs:
                warnings.warn(
                    f'AdamW8bitKahan: ignoring "{key}", which current bitsandbytes '
                    f'no longer supports.')
                kwargs.pop(key)
        super().__init__(*args, **kwargs)
        self.stabilize = stabilize

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        super().init_state(group, p, gindex, pindex)
        self.state[p]['shift'] = self.get_state_buffer(p, dtype=p.dtype)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        # avoid update error from non-contiguous memory layout
        p.data = p.data.contiguous()
        p.grad = p.grad.contiguous()

        state = self.state[p]
        grad = p.grad

        config = self.get_config(gindex, pindex, group)

        state["step"] += 1
        step = state["step"]

        # Percentile clipping only exists on older bitsandbytes, and needs the
        # gnorm_vec buffer that newer versions no longer allocate. Guard on
        # both so this is a no-op (gnorm_scale = 1.0, matching what modern
        # bitsandbytes hardcodes) instead of a KeyError.
        percentile_clipping = config.get("percentile_clipping", 100)
        if percentile_clipping < 100 and "gnorm_vec" in state:
            current_gnorm, clip_value, gnorm_scale = F.percentile_clipping(
                grad,
                state["gnorm_vec"],
                step,
                percentile_clipping,
            )
        else:
            gnorm_scale = 1.0

        shift = state['shift']

        # StableAdamW
        if self.stabilize:
            exp_avg_sq = state['state2']
            eps_sq = torch.tensor(config['eps']**2, dtype=exp_avg_sq.dtype, device=exp_avg_sq.device)
            rms = grad.pow(2).div_(exp_avg_sq.maximum(eps_sq)).mean().sqrt()
            lr = config['lr'] / max(1, rms.item())
        else:
            lr = config['lr']

        if state["state1"].dtype == torch.float:
            F.optimizer_update_32bit(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                lr,
                state["state2"],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config.get("alpha", 0.0),
                config["weight_decay"],
                gnorm_scale,
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )

        elif state["state1"].dtype == torch.uint8 and not config.get("block_wise", True):
            # Legacy non-blockwise 8-bit path. Only reachable on older
            # bitsandbytes; modern versions removed both the config key and
            # the max1/max2/new_max1/new_max2 state buffers this needs.
            F.optimizer_update_8bit(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["eps"],
                step,
                lr,
                state["qmap1"],
                state["qmap2"],
                state["max1"],
                state["max2"],
                state["new_max1"],
                state["new_max2"],
                config["weight_decay"],
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
            )

            # swap maxes
            state["max1"], state["new_max1"] = state["new_max1"], state["max1"]
            state["max2"], state["new_max2"] = state["new_max2"], state["max2"]

        elif state["state1"].dtype == torch.uint8:
            # Blockwise 8-bit: the only 8-bit path modern bitsandbytes keeps.
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                grad,
                shift,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["betas"][2] if len(config["betas"]) >= 3 else 0.0,
                config.get("alpha", 0.0),
                config["eps"],
                step,
                lr,
                state["qmap1"],
                state["qmap2"],
                state["absmax1"],
                state["absmax2"],
                config["weight_decay"],
                gnorm_scale=gnorm_scale,
                skip_zeros=config["skip_zeros"],
            )

        # Kahan: fold the compensated update into p, carrying the residual
        # (the part that fell below bf16 precision) forward in `shift`.
        buffer = p.clone()
        p.add_(shift)
        shift.add_(buffer.sub_(p))