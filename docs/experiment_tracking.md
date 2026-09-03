# Experiment tracking: wandb, Trackio, or none

One tracker is active per run, chosen in the config:

```toml
[monitoring]
tracker = 'wandb'      # 'wandb' | 'trackio' | 'none'
```

TensorBoard and the on-disk PNG sample archive are always written regardless of
this setting — they're local file writes with no network dependency, so they
keep working even when a hosted tracker doesn't.

## Why you might pick each one

| | wandb | Trackio | none |
|---|---|---|---|
| Hosted dashboard | yes | via HF Spaces, or self-host | – |
| Works with no network at all | offline mode, sync later | yes, local dashboard | yes |
| Validation sample images | yes | yes | – |
| Maturity | stable | pre-release | – |

**Trackio** is Hugging Face's local-first tracker. It's API-compatible with
wandb for `init`/`log`/`finish`/`Image`, which is why both fit behind one
interface here. Its local mode needs no network whatsoever, which makes it the
practical option on hosts where `api.wandb.ai` is unreachable — W&B has a
long-standing edge block affecting some regions (see wandb issues #10206 and
#10326) that returns an HTML `403 Forbidden` rather than an auth error, so it
fails identically no matter which API key you use.

## wandb options

```toml
[monitoring]
tracker = 'wandb'
wandb_tracker_name = 'my-project'     # project
wandb_run_name = 'run-1'
wandb_api_key = ''                    # falls back to $WANDB_API_KEY, then netrc
wandb_entity = ''                     # optional team/org
wandb_mode = 'online'                 # 'online' | 'offline' | 'disabled'
wandb_base_url = ''                   # self-hosted W&B, or $WANDB_BASE_URL
wandb_offline_on_failure = true       # if online setup fails, retry offline
```

`wandb_mode = 'offline'` writes a complete run to disk that you upload later
with `wandb sync <run_dir>`. Because a partially written offline run syncs
fine, you can pull the run directory to a machine with access and sync
mid-training for near-live monitoring.

Note the API key format change: W&B now issues ~86-character keys, and SDKs
older than v0.22.3 reject them with `API key must be 40 characters long`. That
is a *different* failure from the region block above — it raises a length
error, not a 403.

## Trackio options

```toml
[monitoring]
tracker = 'trackio'
wandb_tracker_name = 'my-project'     # reused as the Trackio project name
wandb_run_name = 'run-1'
trackio_dir = ''                      # default: <run_dir>/trackio
trackio_space_id = ''                 # e.g. 'username/my-dashboard' (HF Spaces)
trackio_server_url = ''               # e.g. 'http://my-host:7860' (self-hosted)
```

Three deployment modes:

**Local (default).** SQLite plus a Gradio dashboard on the training machine.
No network at all. View it with:

```bash
trackio show --project "my-project"
```

On a remote box, tunnel the port from your own machine:

```bash
ssh -L 7860:localhost:7860 user@training-host
```

Then open `http://localhost:7860`. This gives live metrics *and* live
validation sample images without depending on any third-party service being
reachable from the training host.

**Hugging Face Spaces.** Set `trackio_space_id`; Trackio deploys/uses a free
Space. Requires `huggingface-cli login` with a write-scoped token. Useful when
several people need the dashboard, and viable on hosts that can reach
huggingface.co even if they can't reach api.wandb.ai.

**Self-hosted.** Set `trackio_server_url` to a Trackio server you run. Best
option when you want a shared dashboard but don't want to depend on any
external provider's regional availability.

`trackio_dir` defaults to `<run_dir>/trackio` rather than the global
`~/.cache/huggingface/trackio`, so each run's data stays with its outputs.

## Failure behaviour

Tracking must never take down a training run. That's the central guarantee of
`utils/tracking.py`:

- If a backend fails to start, the run continues with tracking disabled. A
  wandb 403 used to kill jobs before step 1.
- With `wandb_offline_on_failure = true` (the default), a failed online start
  retries offline before giving up, so you keep the data.
- If `log()` fails mid-run, the tracker prints one warning, disables itself,
  and training continues. It is not retried every step for the rest of the run.
- An unknown `tracker` value fails immediately at startup rather than 40
  minutes into latent caching.

## Adding another backend

Subclass `Tracker` in `utils/tracking.py` and implement `init`, `_log`,
`Image`, and `finish`, then add it to `resolve_tracker_name` and
`build_tracker`. Nothing outside that module imports a tracking library
directly. Two rules the existing backends follow:

1. `log()` must forward an explicit `step`. Both wandb and Trackio otherwise
   advance an internal counter once per call, which silently desynchronises the
   logged x-axis from the real training step whenever more than one call
   happens per step (eval, sampling, epoch boundaries).
2. `Image()` returns `None` if the backend can't wrap an image; callers drop
   those rather than treating it as an error.

## Backward compatibility

Configs written before `tracker` existed keep working: if `tracker` is absent,
`enable_wandb = true` selects wandb and `false` selects none. If both are
present, `tracker` wins.