"""Unified experiment-tracking interface over wandb, Trackio, or nothing.

One tracker is active per run, selected with [monitoring] tracker = 'wandb' |
'trackio' | 'none'. The rest of the codebase talks to the Tracker object
returned by build_tracker() and never imports wandb or trackio directly, so
adding a third backend later means adding one subclass here.

Why an abstraction rather than `import trackio as wandb`: Trackio is API
compatible for init/log/finish/Image, but its init() takes a different set of
keyword arguments (no dir=, no mode=, no entity=; instead space_id/server_url),
and it has no login() at all. Those differences live here instead of leaking
into train.py.

Design rules, both learned the hard way on this fork:
  * Tracking must NEVER take down a training run. Every call into a backend is
    wrapped; a failure disables tracking for the rest of the run and training
    continues. A 403 from a monitoring service used to kill jobs before step 1.
  * log() always passes an explicit step. Both backends advance an internal
    counter once per call otherwise, which silently desynchronises the logged
    x-axis from the real training step.
"""

import os


class Tracker:
    """Base interface. NullTracker is the do-nothing implementation."""

    name = 'none'

    def __init__(self):
        self.enabled = False
        self._failed = False

    # -- lifecycle -------------------------------------------------------

    def init(self, project, run_name, config, run_dir):
        return False

    def finish(self):
        pass

    # -- logging ---------------------------------------------------------

    def log(self, data, step=None):
        """Best-effort metric logging. Never raises."""
        if not self.enabled or self._failed:
            return
        try:
            self._log(data, step)
        except Exception as e:
            self._disable_after_failure(e)

    def _log(self, data, step):
        raise NotImplementedError

    def Image(self, image, caption=None):
        """Wrap a PIL image for logging. Returns None if unsupported, which
        callers treat as 'skip the image' rather than an error."""
        return None

    # -- internals -------------------------------------------------------

    def _disable_after_failure(self, exc):
        self._failed = True
        self.enabled = False
        print(f'Warning: {self.name} logging failed ({type(exc).__name__}: {exc}). '
              f'Disabling {self.name} for the rest of this run; training continues '
              'and TensorBoard logging is unaffected.')


class NullTracker(Tracker):
    name = 'none'


class WandbTracker(Tracker):
    name = 'wandb'

    def __init__(self, monitoring):
        super().__init__()
        self.monitoring = monitoring
        self.wandb = None

    def init(self, project, run_name, config, run_dir):
        import wandb
        self.wandb = wandb
        m = self.monitoring

        api_key = m.get('wandb_api_key') or os.environ.get('WANDB_API_KEY') or None
        entity = m.get('wandb_entity') or None
        base_url = m.get('wandb_base_url') or os.environ.get('WANDB_BASE_URL') or None
        mode = m.get('wandb_mode', 'online')
        fallback = m.get('wandb_offline_on_failure', True)

        if base_url:
            os.environ['WANDB_BASE_URL'] = base_url

        def _start(m_):
            # login() only for an online run with an explicit key: it hits the
            # network and is the most common place a run dies before training.
            # Offline needs no auth, and with no key wandb falls back to
            # WANDB_API_KEY / netrc on its own.
            if m_ == 'online' and api_key:
                self.wandb.login(key=api_key)
            self.wandb.init(project=project, name=run_name, entity=entity,
                            config=config, dir=run_dir, mode=m_)

        try:
            _start(mode)
            self.enabled = True
        except Exception as e:
            print(f'Warning: wandb setup failed ({type(e).__name__}: {e})')
            if fallback and mode == 'online':
                print('Warning: retrying wandb in OFFLINE mode; sync later with '
                      f'`wandb sync {run_dir}`')
                try:
                    _start('offline')
                    self.enabled = True
                except Exception as e2:
                    print(f'Warning: offline wandb also failed ({type(e2).__name__}: {e2}); '
                          'continuing with tracking disabled.')
            else:
                print('Warning: continuing with tracking disabled.')
        return self.enabled

    def _log(self, data, step):
        self.wandb.log(data, step=step)

    def Image(self, image, caption=None):
        try:
            return self.wandb.Image(image, caption=caption)
        except Exception:
            return None

    def finish(self):
        if self.enabled and self.wandb is not None:
            try:
                self.wandb.finish()
            except Exception:
                pass


class TrackioTracker(Tracker):
    """Trackio backend.

    Three deployment modes, chosen by config:
      * local (default)  -- SQLite + Gradio dashboard on the training machine.
        No network at all, so it works on hosts where a hosted tracker is
        unreachable. View it with `trackio show`, tunnelling the port if the
        box is remote.
      * space_id         -- free hosted dashboard on Hugging Face Spaces.
      * server_url       -- a Trackio server you run yourself.
    """

    name = 'trackio'

    def __init__(self, monitoring):
        super().__init__()
        self.monitoring = monitoring
        self.trackio = None

    def init(self, project, run_name, config, run_dir):
        try:
            import trackio
        except ImportError:
            print('Warning: tracker = "trackio" but the trackio package is not '
                  'installed (pip install trackio). Continuing without tracking.')
            return False
        self.trackio = trackio
        m = self.monitoring

        space_id = m.get('trackio_space_id') or os.environ.get('TRACKIO_SPACE_ID') or None
        server_url = m.get('trackio_server_url') or os.environ.get('TRACKIO_SERVER_URL') or None
        # Keep run data beside the run's other outputs rather than in the
        # global ~/.cache/huggingface/trackio, so a run is self-contained.
        # trackio.init() has no dir= parameter; TRACKIO_DIR is the supported way.
        trackio_dir = m.get('trackio_dir') or (os.path.join(run_dir, 'trackio') if run_dir else None)
        if trackio_dir:
            os.makedirs(trackio_dir, exist_ok=True)
            os.environ['TRACKIO_DIR'] = trackio_dir

        kwargs = {'project': project, 'name': run_name, 'config': config}
        if space_id:
            kwargs['space_id'] = space_id
        if server_url:
            kwargs['server_url'] = server_url

        try:
            trackio.init(**kwargs)
            self.enabled = True
            where = (f'Hugging Face Space "{space_id}"' if space_id
                     else f'server {server_url}' if server_url
                     else f'locally at {trackio_dir} (view with: trackio show --project "{project}")')
            print(f'Trackio tracking to {where}')
        except Exception as e:
            print(f'Warning: trackio setup failed ({type(e).__name__}: {e}); '
                  'continuing with tracking disabled.')
        return self.enabled

    def _log(self, data, step):
        self.trackio.log(data, step=step)

    def Image(self, image, caption=None):
        try:
            return self.trackio.Image(image, caption=caption)
        except Exception:
            return None

    def finish(self):
        if self.enabled and self.trackio is not None:
            try:
                self.trackio.finish()
            except Exception:
                pass


def resolve_tracker_name(config):
    """Which backend the config asks for.

    [monitoring] tracker = 'wandb' | 'trackio' | 'none' is the current option.
    enable_wandb is still honoured when tracker is absent, so existing configs
    keep working unchanged.
    """
    monitoring = config.get('monitoring') or {}
    tracker = monitoring.get('tracker')
    if tracker is None:
        return 'wandb' if monitoring.get('enable_wandb', False) else 'none'
    tracker = str(tracker).strip().lower()
    if tracker in ('', 'none', 'off', 'disabled', 'false'):
        return 'none'
    if tracker not in ('wandb', 'trackio'):
        raise ValueError(
            f"[monitoring] tracker must be 'wandb', 'trackio', or 'none', got {tracker!r}")
    return tracker


def build_tracker(config, run_dir, is_main=True):
    """Construct and initialise the configured tracker.

    Only the main process tracks; other ranks get a NullTracker so they can
    call tracker.log() unconditionally without duplicating metrics.
    """
    name = resolve_tracker_name(config)
    if name == 'none' or not is_main:
        return NullTracker()

    monitoring = config.get('monitoring') or {}
    project = monitoring.get('wandb_tracker_name') or monitoring.get('project') or 'diffusion-pipe'
    run_name = monitoring.get('wandb_run_name') or monitoring.get('run_name') or None

    tracker = WandbTracker(monitoring) if name == 'wandb' else TrackioTracker(monitoring)
    tracker.init(project, run_name, config, run_dir)
    return tracker