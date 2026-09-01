"""Inference validation sampling: generate images from fixed prompts during
training and log them to wandb / TensorBoard / disk.

This module owns everything that is NOT model-specific:

  * config parsing and the per-prompt override merge
  * the flow-matching sigma schedule and Euler integration
  * seed policy (fixed-per-prompt by default, so epoch-over-epoch differences
    are attributable to training rather than to a different noise draw)
  * cadence gating (every N steps / epochs / before the first step)
  * logging to wandb, TensorBoard, and a tool-independent PNG archive on disk

Model-specific behaviour lives behind the `SamplingAdapter` interface that each
BasePipeline subclass returns from `get_sampling_adapter()`. A model only has
to say how to encode prompts, how to assemble the input tuple its layer stack
expects, and how to turn final latents back into pixels.

Design notes
------------
Flow convention: this codebase trains with ``x_t = (1-t)*x_1 + t*x_0`` where
``x_1`` is the clean latent and ``x_0`` is noise, and the target is
``x_0 - x_1``. So t=0 is clean, t=1 is pure noise, and the model predicts a
velocity pointing from clean toward noise. Sampling therefore starts at t=1
and integrates *down* to t=0 with ``x += (sigma_next - sigma) * v``.

CFG: neither Mage-Flow nor Anima has a distilled guidance embedding (unlike
Flux/Chroma, which take a `guidance` vector as a model input), so guidance is
classic dual-forward CFG: ``v = v_uncond + cfg * (v_cond - v_uncond)``. A cfg
of <= 1.0 skips the negative pass entirely, which halves sampling cost.
"""

import time
from pathlib import Path

import torch


# Default generation settings. Every one of these can be overridden globally in
# the [sampling] config table, and again per-prompt in [[sampling.prompts]].
DEFAULT_SAMPLE_SETTINGS = {
    'steps': 30,
    'cfg': 5.0,
    'shift': 6.0,          # inference-time flow shift; independent of training shift
    'width': 1024,
    'height': 1024,
    'negative_prompt': '',
    'seed': 42,
    'renormalize_cfg': False,
}

# Settings that may be overridden per-prompt. Anything outside this set is a
# run-level concern (cadence, logging) and is rejected inside a prompt entry so
# typos surface at startup instead of being silently ignored.
PER_PROMPT_KEYS = frozenset(DEFAULT_SAMPLE_SETTINGS) | {'prompt', 'label'}

# Run-level keys valid in the [sampling] table.
RUN_LEVEL_KEYS = frozenset({
    'enable', 'prompts',
    'sample_every_n_epochs', 'sample_every_n_steps', 'sample_at_first',
    'seed_strategy', 'save_to_disk', 'log_to_wandb', 'log_to_tensorboard',
    'batch_size',
}) | frozenset(DEFAULT_SAMPLE_SETTINGS)


class SamplingConfigError(ValueError):
    """Raised for malformed [sampling] config, at startup rather than mid-run."""


# ---------------------------------------------------------------------------
# Config parsing / per-prompt override merge
# ---------------------------------------------------------------------------

def _coerce_prompt_entry(entry, index):
    """Normalize one prompt entry to a dict. A bare string is shorthand for
    {'prompt': <string>} with everything else inherited."""
    if isinstance(entry, str):
        return {'prompt': entry}
    if isinstance(entry, dict):
        if 'prompt' not in entry:
            raise SamplingConfigError(
                f'sampling.prompts[{index}] has no "prompt" key: {entry!r}')
        return dict(entry)
    raise SamplingConfigError(
        f'sampling.prompts[{index}] must be a string or a table, got {type(entry).__name__}')


def _validate_settings(settings, where):
    """Range/type-check a fully merged settings dict."""
    steps = settings['steps']
    if not isinstance(steps, int) or steps < 1:
        raise SamplingConfigError(f'{where}: steps must be an integer >= 1, got {steps!r}')
    cfg = settings['cfg']
    if not isinstance(cfg, (int, float)) or cfg < 0:
        raise SamplingConfigError(f'{where}: cfg must be a number >= 0, got {cfg!r}')
    shift = settings['shift']
    if not isinstance(shift, (int, float)) or shift <= 0:
        raise SamplingConfigError(f'{where}: shift must be a number > 0, got {shift!r}')
    for dim in ('width', 'height'):
        val = settings[dim]
        if not isinstance(val, int) or val < 16:
            raise SamplingConfigError(f'{where}: {dim} must be an integer >= 16, got {val!r}')
        if val % 16 != 0:
            raise SamplingConfigError(
                f'{where}: {dim} must be a multiple of 16 (got {val}); '
                f'nearest valid values are {(val // 16) * 16} and {((val // 16) + 1) * 16}')
    if not isinstance(settings['seed'], int):
        raise SamplingConfigError(f'{where}: seed must be an integer, got {settings["seed"]!r}')
    if not isinstance(settings['negative_prompt'], str):
        raise SamplingConfigError(
            f'{where}: negative_prompt must be a string, got {type(settings["negative_prompt"]).__name__}')


def build_sampling_config(config):
    """Parse the top-level [sampling] table into a normalized config dict.

    Returns None when sampling is not configured or explicitly disabled, so
    callers can treat "no sampling" as a cheap early-out.

    The merge rule is: start from DEFAULT_SAMPLE_SETTINGS, overlay the
    run-level [sampling] values, then overlay each prompt's own values. A
    per-prompt key wins over the run-level value, which wins over the default.
    The merge is deliberately shallow — every setting is a scalar.
    """
    sampling = config.get('sampling')
    if not sampling:
        return None
    if not isinstance(sampling, dict):
        raise SamplingConfigError('[sampling] must be a table')
    if not sampling.get('enable', True):
        return None

    unknown = set(sampling) - RUN_LEVEL_KEYS
    if unknown:
        raise SamplingConfigError(
            f'unknown key(s) in [sampling]: {sorted(unknown)}. '
            f'Valid keys: {sorted(RUN_LEVEL_KEYS)}')

    raw_prompts = sampling.get('prompts')
    if not raw_prompts:
        raise SamplingConfigError(
            '[sampling] is enabled but has no prompts. Add sampling.prompts.')
    if isinstance(raw_prompts, str):
        raw_prompts = [raw_prompts]

    # Run-level defaults: DEFAULT_SAMPLE_SETTINGS overlaid with [sampling] values.
    shared = dict(DEFAULT_SAMPLE_SETTINGS)
    for key in DEFAULT_SAMPLE_SETTINGS:
        if key in sampling:
            shared[key] = sampling[key]
    _validate_settings(shared, '[sampling]')

    seed_strategy = sampling.get('seed_strategy', 'fixed')
    if seed_strategy not in ('fixed', 'walk'):
        raise SamplingConfigError(
            f"sampling.seed_strategy must be 'fixed' or 'walk', got {seed_strategy!r}")

    prompts = []
    for i, raw in enumerate(raw_prompts):
        entry = _coerce_prompt_entry(raw, i)
        unknown = set(entry) - PER_PROMPT_KEYS
        if unknown:
            raise SamplingConfigError(
                f'unknown key(s) in sampling.prompts[{i}]: {sorted(unknown)}. '
                f'Valid per-prompt keys: {sorted(PER_PROMPT_KEYS)}')
        merged = dict(shared)
        for key in DEFAULT_SAMPLE_SETTINGS:
            if key in entry:
                merged[key] = entry[key]
        merged['prompt'] = entry['prompt']
        # A label is only for display/filenames; default to a truncated prompt.
        merged['label'] = entry.get('label') or _default_label(entry['prompt'], i)
        merged['index'] = i
        _validate_settings(merged, f'sampling.prompts[{i}]')
        prompts.append(merged)

    every_n_epochs = sampling.get('sample_every_n_epochs', 1)
    every_n_steps = sampling.get('sample_every_n_steps', None)
    if every_n_epochs is not None and (not isinstance(every_n_epochs, int) or every_n_epochs < 0):
        raise SamplingConfigError('sampling.sample_every_n_epochs must be a non-negative integer')
    if every_n_steps is not None and (not isinstance(every_n_steps, int) or every_n_steps < 0):
        raise SamplingConfigError('sampling.sample_every_n_steps must be a non-negative integer')
    if not every_n_epochs and not every_n_steps and not sampling.get('sample_at_first', False):
        raise SamplingConfigError(
            '[sampling] is enabled but no cadence is set. Set sample_every_n_epochs, '
            'sample_every_n_steps, or sample_at_first.')

    batch_size = sampling.get('batch_size', 1)
    if not isinstance(batch_size, int) or batch_size < 1:
        raise SamplingConfigError('sampling.batch_size must be an integer >= 1')

    return {
        'prompts': prompts,
        'shared': shared,
        'sample_every_n_epochs': every_n_epochs,
        'sample_every_n_steps': every_n_steps,
        'sample_at_first': sampling.get('sample_at_first', False),
        'seed_strategy': seed_strategy,
        'save_to_disk': sampling.get('save_to_disk', True),
        'log_to_wandb': sampling.get('log_to_wandb', True),
        'log_to_tensorboard': sampling.get('log_to_tensorboard', True),
        'batch_size': batch_size,
    }


def _default_label(prompt, index):
    """Short, filesystem-safe label derived from the prompt."""
    cleaned = ''.join(c if (c.isalnum() or c in ' -_') else '' for c in prompt).strip()
    cleaned = '_'.join(cleaned.split())[:40].strip('_')
    return cleaned or f'prompt{index}'


def should_sample(sampling_config, step, epoch, finished_epoch):
    """Cadence gate. Mirrors the eval cadence in train.py: a step-based trigger,
    or an epoch-based trigger that only fires on the step where an epoch ends."""
    if sampling_config is None:
        return False
    every_n_steps = sampling_config['sample_every_n_steps']
    if every_n_steps and step % every_n_steps == 0:
        return True
    every_n_epochs = sampling_config['sample_every_n_epochs']
    if finished_epoch and every_n_epochs and epoch % every_n_epochs == 0:
        return True
    return False


def resolve_seed(prompt_settings, sampling_config, round_index):
    """Seed for one prompt in one sampling round.

    'fixed' (default) reuses the same seed every round so identical starting
    noise makes epoch-over-epoch comparison meaningful — this is the whole
    point of the feature. 'walk' advances the seed per round for variety.
    """
    base = prompt_settings['seed']
    if sampling_config['seed_strategy'] == 'walk':
        return base + round_index
    return base


# ---------------------------------------------------------------------------
# Flow-matching schedule
# ---------------------------------------------------------------------------

def build_sigmas(steps, shift, device=None, dtype=torch.float32):
    """Static-shift flow-matching sigma schedule, descending from ~1 to 0.

    Reproduces the reference Mage-Flow pipeline exactly: base sigmas
    ``linspace(1, 1/steps, steps)``, each mapped through the static shift
    ``shift*s / (1 + (shift-1)*s)``, with a terminal 0 appended. Returns
    ``steps + 1`` values so the Euler loop can read ``sigmas[i]`` and
    ``sigmas[i+1]`` for every step.
    """
    if steps < 1:
        raise ValueError(f'steps must be >= 1, got {steps}')
    base = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    if shift and shift != 1.0:
        base = (shift * base) / (1.0 + (shift - 1.0) * base)
    sigmas = torch.cat([base, torch.zeros(1, dtype=torch.float64)])
    return sigmas.to(device=device, dtype=dtype)


def euler_step(latents, velocity, sigma, sigma_next):
    """One Euler integration step along the flow, from sigma toward sigma_next.

    With ``x_t = (1-t)*x_clean + t*noise`` the velocity is ``d x_t / dt``, so
    stepping to a lower sigma is ``x += (sigma_next - sigma) * v``. sigma_next
    is below sigma, making the update subtract velocity as noise is removed.
    """
    dt = sigma_next - sigma
    return latents + dt * velocity.to(latents.dtype)


def combine_cfg(cond, uncond, cfg, renormalize=False):
    """Classic dual-forward classifier-free guidance.

    Optional renormalization rescales the guided velocity per token back to the
    conditional velocity's norm, which reduces oversaturation at high cfg.
    """
    combined = uncond + cfg * (cond - uncond)
    if not renormalize:
        return combined
    cond_norm = torch.norm(cond, dim=-1, keepdim=True)
    comb_norm = torch.norm(combined, dim=-1, keepdim=True)
    return combined * (cond_norm / (comb_norm + 1e-6))


def uses_negative_branch(cfg, negative_prompt):
    """Whether a sample needs the uncond forward pass at all.

    Matches the reference pipeline: guidance below or at 1.0 is a no-op, so the
    negative pass is skipped and sampling costs one forward per step instead of
    two. An empty negative prompt still counts as a valid uncond conditioning
    (it is encoded as a blank prompt), so only cfg gates this.
    """
    return cfg > 1.0


# ---------------------------------------------------------------------------
# Per-model interface
# ---------------------------------------------------------------------------

class SamplingAdapter:
    """What a model must provide for validation sampling.

    Everything else — schedule, CFG, seeding, cadence, logging — is handled by
    this module, so adding a new model means implementing these four methods
    and nothing more.

    The latent representation is opaque to the driver: it only ever passes
    latents back into `predict_velocity` and `decode`, and integrates them with
    elementwise arithmetic. A model is free to use [B, L, C] token sequences
    (Mage-Flow) or [B, C, F, H, W] volumes (Anima) without the driver caring.
    """

    #: Set False by a model that cannot currently sample (e.g. missing VAE).
    supported = True

    def encode_prompt(self, prompt):
        """Return opaque text conditioning for one prompt string.

        Called once per distinct prompt at startup, while the text encoder is
        still loaded, and the result is reused for every sampling round.
        """
        raise NotImplementedError

    def init_latents(self, width, height, generator, device, dtype):
        """Pure-noise starting latents for the requested pixel size."""
        raise NotImplementedError

    def predict_velocity(self, latents, text_cond, sigma, width, height):
        """Run the model's layer stack once and return the predicted velocity,
        the same shape as `latents`."""
        raise NotImplementedError

    def decode(self, latents, width, height):
        """Decode final latents to a PIL image."""
        raise NotImplementedError


def run_layer_stack(layers, inputs):
    """Run a model's `to_layers()` stack sequentially and return its output.

    This is the same forward path training uses, so samples reflect exactly
    what is being trained — including block-swap offloading, which the layer
    wrappers drive themselves. It is deliberately not DeepSpeed's pipeline
    engine: that is dataloader-driven and splits layers across ranks.
    """
    x = inputs
    for layer in layers:
        x = layer(x)
    return x


# ---------------------------------------------------------------------------
# Sampling driver
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_one(adapter, prompt_settings, seed, device, dtype):
    """Generate a single image. Returns a PIL image."""
    width, height = prompt_settings['width'], prompt_settings['height']
    steps = prompt_settings['steps']
    cfg = prompt_settings['cfg']

    generator = torch.Generator(device='cpu').manual_seed(seed)
    latents = adapter.init_latents(width, height, generator, device, dtype)
    sigmas = build_sigmas(steps, prompt_settings['shift'], device=device, dtype=torch.float32)

    cond = adapter.encode_prompt(prompt_settings['prompt'])
    need_uncond = uses_negative_branch(cfg, prompt_settings['negative_prompt'])
    uncond = adapter.encode_prompt(prompt_settings['negative_prompt']) if need_uncond else None

    for i in range(steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        v = adapter.predict_velocity(latents, cond, sigma, width, height)
        if need_uncond:
            v_unc = adapter.predict_velocity(latents, uncond, sigma, width, height)
            v = combine_cfg(v, v_unc, cfg, renormalize=prompt_settings['renormalize_cfg'])
        latents = euler_step(latents, v, sigma, sigma_next)

    return adapter.decode(latents, width, height)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _caption_for(prompt_settings, seed):
    p = prompt_settings
    return (f"{p['prompt']}\n"
            f"steps={p['steps']} cfg={p['cfg']} shift={p['shift']} "
            f"{p['width']}x{p['height']} seed={seed}")


def _annotate(image, text):
    """Stamp a caption strip under an image.

    wandb captions images natively; TensorBoard's image viewer does not, so the
    text is burned in to keep the on-disk archive and the TB gallery
    self-describing.
    """
    from PIL import Image, ImageDraw
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None

    lines, width = [], image.width
    for raw in text.split('\n'):
        # crude wrap at ~7px/char for the default bitmap font
        limit = max(16, width // 7)
        while len(raw) > limit:
            lines.append(raw[:limit])
            raw = raw[limit:]
        lines.append(raw)

    strip = 4 + 12 * len(lines)
    out = Image.new('RGB', (width, image.height + strip), (16, 16, 16))
    out.paste(image, (0, 0))
    draw = ImageDraw.Draw(out)
    for i, line in enumerate(lines):
        draw.text((4, image.height + 2 + 12 * i), line, fill=(230, 230, 230), font=font)
    return out


def log_samples(results, sampling_config, tb_writer, step, run_dir, label,
                wandb_module=None):
    """Write generated samples to wandb, TensorBoard, and disk.

    `results` is a list of (prompt_settings, seed, PIL image). Every sink is
    best-effort and independently guarded: a wandb hiccup must never take down
    a training run that has been going for hours.
    """
    import numpy as np

    if sampling_config['save_to_disk'] and run_dir is not None:
        out_dir = Path(run_dir) / 'samples' / label
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            for settings, seed, image in results:
                image.save(out_dir / f"{settings['index']:02d}_{settings['label']}_seed{seed}.png")
        except Exception as e:
            print(f'Warning: failed to save sample images to disk: {e}')

    if sampling_config['log_to_wandb'] and wandb_module is not None:
        try:
            images = [
                wandb_module.Image(image, caption=_caption_for(settings, seed))
                for settings, seed, image in results
            ]
            wandb_module.log({'samples': images, 'step': step})
        except Exception as e:
            print(f'Warning: failed to log samples to wandb: {e}')

    if sampling_config['log_to_tensorboard'] and tb_writer is not None:
        try:
            for settings, seed, image in results:
                annotated = _annotate(image, _caption_for(settings, seed))
                arr = np.array(annotated).transpose(2, 0, 1)  # HWC -> CHW
                tag = f"samples/{settings['index']:02d}_{settings['label']}"
                tb_writer.add_image(tag, arr, step)
        except Exception as e:
            print(f'Warning: failed to log samples to TensorBoard: {e}')


# ---------------------------------------------------------------------------
# Entry point called from the training loop
# ---------------------------------------------------------------------------

def generate_and_log_samples(model, sampling_config, tb_writer, step, run_dir,
                             label, round_index, wandb_module=None,
                             disable_block_swap=False):
    """Generate every configured prompt and log the results.

    Mirrors `evaluate()` in train.py: empty the cache, put block swapping into
    inference mode, isolate RNG so sampling can never perturb the training
    noise stream, then restore training mode. Sampling is synchronous, so
    training simply pauses for the duration.
    """
    from utils.isolate_rng import isolate_rng

    adapter = model.get_sampling_adapter()
    if adapter is None or not adapter.supported:
        return

    prompts = sampling_config['prompts']
    print(f'Generating {len(prompts)} validation sample(s) [{label}]')
    start = time.time()

    empty_cache = getattr(torch.cuda, 'empty_cache', lambda: None)
    empty_cache()
    model.prepare_block_swap_inference(disable_block_swap=disable_block_swap)

    results = []
    try:
        with torch.no_grad(), isolate_rng():
            device, dtype = model.sampling_device(), model.sampling_dtype()
            for settings in prompts:
                seed = resolve_seed(settings, sampling_config, round_index)
                try:
                    image = sample_one(adapter, settings, seed, device=device, dtype=dtype)
                    results.append((settings, seed, image))
                except Exception as e:
                    # One bad prompt shouldn't abort the whole round, let alone
                    # the training run.
                    print(f"Warning: sampling failed for prompt "
                          f"{settings['index']} ({settings['label']!r}): {type(e).__name__}: {e}")
    finally:
        empty_cache()
        model.prepare_block_swap_training()

    duration = time.time() - start
    if results:
        log_samples(results, sampling_config, tb_writer, step, run_dir, label,
                    wandb_module=wandb_module)
    print(f'Generated {len(results)}/{len(prompts)} sample(s) in {duration:.1f}s')

    if tb_writer is not None:
        try:
            tb_writer.add_scalar('samples/sample_time_sec', duration, step)
        except Exception:
            pass
    if wandb_module is not None and sampling_config['log_to_wandb']:
        try:
            wandb_module.log({'samples/sample_time_sec': duration, 'step': step})
        except Exception:
            pass
