# Inference validation sampling

Generate images from a fixed set of prompts during training and log them to
Weights & Biases, TensorBoard, and disk. Watching samples evolve across epochs
is how you build the instinct for where a run is heading — loss alone tells you
almost nothing about whether a style is actually landing.

Supported models: `mage_flow`, `anima`.

## Quick start

Add a `[sampling]` table to your training TOML:

```toml
[sampling]
sample_every_n_epochs = 1
sample_at_first = true          # baseline before any training

steps = 30
cfg = 5.0
shift = 6.0
width = 1024
height = 1024
negative_prompt = ''
seed = 42

prompts = [
    'Drawn by hyatsu, 1girl, solo, long hair, looking at viewer',
    'Drawn by hyatsu, 1girl, outdoors, scenery, wide shot',
]
```

Samples appear in wandb under the `samples` key as a gallery, in TensorBoard
under `samples/*`, and on disk at `<output_dir>/<run>/samples/<label>/`.

## Per-prompt overrides

Any generation setting can be set once at the `[sampling]` level and overridden
per prompt. A bare string inherits everything; a table overrides only the keys
it names. This is a shallow merge — per-prompt beats run-level beats default.

```toml
[sampling]
steps = 30
cfg = 5.0
shift = 6.0

[[sampling.prompts]]
prompt = 'Drawn by hyatsu, 1girl, portrait'
# inherits steps=30, cfg=5.0, shift=6.0

[[sampling.prompts]]
prompt = 'Drawn by hyatsu, 1girl, full body, standing'
height = 1536                    # taller frame, everything else inherited
width = 1024

[[sampling.prompts]]
prompt = 'Drawn by hyatsu, wide landscape, no humans'
shift = 3.0                      # probe shift sensitivity on one prompt only
label = 'shift_probe'

[[sampling.prompts]]
prompt = 'a photo of a cat'      # off-distribution control
cfg = 1.0                        # no guidance -> half the compute
steps = 20
seed = 7
```

Unknown keys are rejected at startup rather than silently ignored, so a typo
like `stpes = 20` fails immediately instead of quietly doing nothing for the
whole run.

## Settings

| Key | Default | Scope | Meaning |
|---|---|---|---|
| `steps` | 30 | both | denoising steps |
| `cfg` | 5.0 | both | guidance scale; `<= 1.0` skips the negative pass entirely |
| `shift` | 6.0 | both | **inference** flow shift, independent of the training `shift` |
| `width` / `height` | 1024 | both | output size; must be multiples of 16 |
| `negative_prompt` | `''` | both | only used when `cfg > 1.0` |
| `seed` | 42 | both | see seeding below |
| `renormalize_cfg` | false | both | rescale guided velocity to the conditional's norm; reduces oversaturation at high cfg |
| `label` | derived from prompt | per-prompt | short name used in filenames and TB tags |
| `sample_every_n_epochs` | 1 | run | epoch cadence |
| `sample_every_n_steps` | none | run | step cadence (independent of epochs) |
| `sample_at_first` | false | run | generate a baseline before training starts |
| `seed_strategy` | `'fixed'` | run | `'fixed'` or `'walk'` |
| `save_to_disk` | true | run | write PNGs under `<run>/samples/` |
| `log_to_wandb` | true | run | log a wandb gallery |
| `log_to_tensorboard` | true | run | log to TensorBoard |
| `enable` | true | run | set false to disable without deleting the table |

### Inference shift is not training shift

`[model].shift` controls the timestep distribution you *train* on.
`[sampling].shift` controls the sigma schedule you *sample* with. They serve
different purposes and there is no reason for them to match. Overriding `shift`
on a single prompt is a cheap way to see how sensitive the current checkpoint
is to that choice without running a separate sweep.

### Seeding

`seed_strategy = 'fixed'` (the default) reuses each prompt's seed every round,
so every checkpoint generates from *identical* starting noise. This is the
whole point of the feature: differences you see between epochs are then
attributable to training, not to a luckier noise draw. Use `'walk'` (seed
advances each round) only if you would rather see variety than comparability.

Each prompt can carry its own `seed`, so you can keep a few fixed compositions
and let others vary.

### Cadence

Sampling has its own cadence, separate from `eval_every_n_epochs`. Loss eval is
one forward pass; sampling is a full denoising loop per prompt (doubled when
`cfg > 1`), so most runs want it much less often. You do not need an eval
dataset configured to use sampling.

`sample_at_first = true` gives you the "before" half of every before/after
comparison, and surfaces a broken sampling config in the first minute of a run
instead of at the end of epoch 1.

## Cost and memory

Sampling is synchronous: training pauses while it runs. Rough cost per round is
`prompts x steps x (2 if cfg > 1 else 1)` transformer forward passes at your
chosen resolution.

To keep it cheap, start with 2-4 prompts at `steps = 20-30`. A `cfg = 1.0`
prompt costs half as much as a guided one.

The feature reuses the same block-swap inference/training transitions and CUDA
cache clearing that loss eval already uses, and restores training mode in a
`finally` block so a failed sample can't leave the model in inference mode.
Sample generation is wrapped in `isolate_rng()`, so it cannot perturb the
training noise stream — enabling sampling does not change your training
trajectory.

### The VAE is kept on CPU

Latent caching normally frees the VAE to the `meta` device. Sampling needs it
to decode at every round, so when `[sampling]` is enabled the VAE is parked on
CPU instead and paged to GPU only while decoding. That costs host RAM rather
than VRAM. The same applies whether or not `cache_text_embeddings` is set.

### Prompt embeddings are computed once

All sampling prompts (and negative prompts) are encoded once at startup, before
latent caching runs, and reused for every round. With
`cache_text_embeddings = false` — the usual setup, since per-step tag shuffling
and dropout require it — the text encoder stays resident anyway, but
pre-encoding still avoids re-encoding the same fixed prompts every round. With
`cache_text_embeddings = true` the encoder is freed after caching, and
pre-encoding is what makes sampling work at all.

## Limitations

**`pipeline_stages` must be 1.** `to_layers()` is partitioned across pipeline
stages, so with more than one stage no single rank holds the whole model, and
generating an image needs the full stack. When `pipeline_stages > 1` the
feature prints a warning at startup and disables itself rather than crashing
mid-run. Data-parallel training across multiple GPUs is fine — generation runs
on rank 0 while the other ranks wait at a barrier.

Supporting multi-stage generation would mean passing activations between ranks
manually (`dist.send`/`recv`) or streaming layers from CPU on a single rank.
Neither is implemented, and as far as I know no open-source diffusion trainer
does it today.

**Images only.** Video sampling is not implemented; Anima latents are generated
as single-frame volumes.

## Adding a new model

Implement `get_sampling_adapter()` on the pipeline, returning a
`utils.validation_sampling.SamplingAdapter` with four methods:

```python
class MyModelSamplingAdapter(vsampling.SamplingAdapter):
    def prepare(self, prompt_strings): ...      # optional: pre-encode at startup
    def encode_prompt(self, prompt): ...        # -> opaque text conditioning
    def init_latents(self, width, height, generator, device, dtype): ...
    def predict_velocity(self, latents, text_cond, sigma, width, height): ...
    def decode(self, latents, width, height): ...  # -> PIL.Image
```

Everything else — schedule, CFG, seeding, cadence, logging, error containment —
is handled by the shared module. The latent representation is opaque to the
driver, so a model can use `[B, L, C]` token sequences (Mage-Flow) or
`[B, C, T, H, W]` volumes (Anima) without the driver caring.

Two things to watch for, both of which bit the existing adapters:

1. If your `to_layers()` hands `InitialLayer` a live text encoder in
   on-the-fly caption mode, it will interpret its text inputs as *token ids*.
   Sampling always supplies precomputed *embeddings*, so build a separate layer
   stack with the encoder set to `None`. The wrappers only hold references, so
   weights and the block-swap offloader are still shared.
2. Return velocity in the same shape as the latents you were handed. The driver
   integrates with plain elementwise arithmetic.

## Flow convention

This codebase trains with `x_t = (1-t)*x_clean + t*noise` and target
`noise - x_clean`, so **t=0 is clean and t=1 is pure noise**, and the model
predicts a velocity pointing from clean toward noise. Sampling starts at t=1
and integrates down to t=0 with `x += (sigma_next - sigma) * v`.

The sigma schedule is `linspace(1, 1/steps, steps)` mapped through the static
shift `shift*s / (1 + (shift-1)*s)` with a terminal 0 appended. This reproduces
diffusers' `FlowMatchEulerDiscreteScheduler` exactly (verified bit-for-bit
against it across step counts and shift values), and matches the vendored
reference pipeline in `Mage/mage_flow/pipeline.py`.

Neither Mage-Flow nor Anima has a distilled guidance embedding — unlike Flux
and Chroma in this repo, they take no `guidance` vector as a model input — so
guidance is classic dual-forward CFG:
`v = v_uncond + cfg * (v_cond - v_uncond)`.

## Tests

Logic is covered offline, with no GPU or model weights:

```bash
python3 -m pytest test/test_validation_sampling.py test/test_sampling_driver.py -q
```

These cover the config merge, schedule correctness against diffusers, seed
determinism, CFG forward-pass counts, block-swap restore-on-failure, RNG
isolation, and per-prompt failure containment.
