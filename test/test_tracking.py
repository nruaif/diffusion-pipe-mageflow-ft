"""Tests for utils/tracking.py.

The central guarantee is that tracking can never take down a training run:
every backend failure path must end with a disabled tracker and no exception
escaping. Also covers config selection and backward compatibility with configs
written before the `tracker` key existed.

Run with: python3 -m pytest test/test_tracking.py -q
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import tracking  # noqa: E402


# --- backend selection -----------------------------------------------------

def test_no_monitoring_section_means_none():
    assert tracking.resolve_tracker_name({}) == 'none'


def test_explicit_trackio():
    assert tracking.resolve_tracker_name({'monitoring': {'tracker': 'trackio'}}) == 'trackio'


def test_explicit_wandb():
    assert tracking.resolve_tracker_name({'monitoring': {'tracker': 'wandb'}}) == 'wandb'


@pytest.mark.parametrize('value', ['none', 'off', 'disabled', 'false', ''])
def test_disabled_spellings(value):
    assert tracking.resolve_tracker_name({'monitoring': {'tracker': value}}) == 'none'


def test_tracker_name_is_case_insensitive():
    assert tracking.resolve_tracker_name({'monitoring': {'tracker': 'TrackIO'}}) == 'trackio'


def test_unknown_tracker_raises_at_startup():
    with pytest.raises(ValueError, match='must be'):
        tracking.resolve_tracker_name({'monitoring': {'tracker': 'mlflow'}})


# --- backward compatibility ------------------------------------------------

def test_legacy_enable_wandb_true_still_selects_wandb():
    assert tracking.resolve_tracker_name({'monitoring': {'enable_wandb': True}}) == 'wandb'


def test_legacy_enable_wandb_false_selects_none():
    assert tracking.resolve_tracker_name({'monitoring': {'enable_wandb': False}}) == 'none'


def test_explicit_tracker_overrides_legacy_enable_wandb():
    # A config carrying both should follow the newer, more specific key.
    cfg = {'monitoring': {'enable_wandb': True, 'tracker': 'trackio'}}
    assert tracking.resolve_tracker_name(cfg) == 'trackio'


def test_tracker_none_overrides_legacy_enable_wandb_true():
    cfg = {'monitoring': {'enable_wandb': True, 'tracker': 'none'}}
    assert tracking.resolve_tracker_name(cfg) == 'none'


# --- NullTracker -----------------------------------------------------------

def test_null_tracker_is_disabled_and_silent():
    t = tracking.NullTracker()
    assert t.enabled is False
    t.log({'a': 1}, step=1)      # must not raise
    t.finish()                   # must not raise
    assert t.Image(object()) is None


def test_build_tracker_returns_null_on_non_main_process():
    cfg = {'monitoring': {'tracker': 'trackio'}}
    t = tracking.build_tracker(cfg, run_dir=None, is_main=False)
    assert isinstance(t, tracking.NullTracker)
    assert t.enabled is False


def test_build_tracker_returns_null_when_disabled():
    t = tracking.build_tracker({'monitoring': {'tracker': 'none'}}, run_dir=None)
    assert isinstance(t, tracking.NullTracker)


# --- failure isolation (the property that matters most) --------------------

class _Boom(tracking.Tracker):
    name = 'boom'

    def __init__(self):
        super().__init__()
        self.enabled = True
        self.attempts = 0

    def _log(self, data, step):
        self.attempts += 1
        raise RuntimeError('backend exploded')


def test_log_failure_never_raises():
    t = _Boom()
    t.log({'loss': 1.0}, step=1)   # must not raise


def test_log_failure_disables_tracker():
    t = _Boom()
    t.log({'loss': 1.0}, step=1)
    assert t.enabled is False


def test_log_failure_is_not_retried_every_step():
    # A dead backend must not be hammered (or re-warned) once per step for the
    # rest of a multi-hour run.
    t = _Boom()
    for i in range(50):
        t.log({'loss': 1.0}, step=i)
    assert t.attempts == 1


def test_disabled_tracker_ignores_further_logs():
    t = _Boom()
    t.enabled = False
    t.log({'loss': 1.0}, step=1)
    assert t.attempts == 0


# --- wandb backend, failure paths (no network, mocked module) --------------

class _FakeWandb:
    def __init__(self, fail_online=False, fail_offline=False):
        self.fail_online, self.fail_offline = fail_online, fail_offline
        self.modes = []

    def login(self, key=None):
        if self.fail_online:
            raise Exception('403 Forbidden')

    def init(self, mode=None, **kw):
        self.modes.append(mode)
        if mode == 'online' and self.fail_online:
            raise Exception('403 Forbidden')
        if mode == 'offline' and self.fail_offline:
            raise Exception('disk full')

    def log(self, data, step=None):
        pass

    def Image(self, image, caption=None):
        return ('img', caption)

    def finish(self):
        pass


def _wandb_tracker(monkeypatch, fake, monitoring):
    t = tracking.WandbTracker(monitoring)
    monkeypatch.setitem(sys.modules, 'wandb', fake)
    return t


def test_wandb_online_success(monkeypatch, tmp_path):
    fake = _FakeWandb()
    t = _wandb_tracker(monkeypatch, fake, {'wandb_api_key': 'k'})
    assert t.init('proj', 'run', {}, str(tmp_path)) is True
    assert fake.modes == ['online']


def test_wandb_403_falls_back_to_offline(monkeypatch, tmp_path):
    # The Russia/region-block case: online 403s, offline still works.
    fake = _FakeWandb(fail_online=True)
    t = _wandb_tracker(monkeypatch, fake, {'wandb_api_key': 'k'})
    assert t.init('proj', 'run', {}, str(tmp_path)) is True
    # login() raises before init() is reached on the online attempt, so only
    # the offline init lands -- the point is that the run survives either way.
    assert fake.modes == ['offline']
    assert t.enabled is True


def test_wandb_total_failure_disables_without_raising(monkeypatch, tmp_path):
    fake = _FakeWandb(fail_online=True, fail_offline=True)
    t = _wandb_tracker(monkeypatch, fake, {'wandb_api_key': 'k'})
    assert t.init('proj', 'run', {}, str(tmp_path)) is False
    assert t.enabled is False


def test_wandb_fallback_can_be_disabled(monkeypatch, tmp_path):
    fake = _FakeWandb(fail_online=True)
    t = _wandb_tracker(monkeypatch, fake,
                       {'wandb_api_key': 'k', 'wandb_offline_on_failure': False})
    assert t.init('proj', 'run', {}, str(tmp_path)) is False
    # No offline retry attempted.
    assert fake.modes == []


def test_wandb_offline_mode_skips_login(monkeypatch, tmp_path):
    # login() is the usual place a blocked network kills a run; offline needs
    # no auth, so it must not be called.
    fake = _FakeWandb(fail_online=True)   # login would raise if called
    t = _wandb_tracker(monkeypatch, fake,
                       {'wandb_api_key': 'k', 'wandb_mode': 'offline'})
    assert t.init('proj', 'run', {}, str(tmp_path)) is True
    assert fake.modes == ['offline']


# --- trackio backend -------------------------------------------------------

class _FakeTrackio:
    def __init__(self, fail=False):
        self.fail, self.kwargs = fail, None

    def init(self, **kw):
        self.kwargs = kw
        if self.fail:
            raise Exception('nope')

    def log(self, data, step=None):
        pass

    def Image(self, image, caption=None):
        return ('img', caption)

    def finish(self):
        pass


def test_trackio_local_mode_sets_dir_and_passes_no_remote_kwargs(monkeypatch, tmp_path):
    fake = _FakeTrackio()
    monkeypatch.setitem(sys.modules, 'trackio', fake)
    t = tracking.TrackioTracker({})
    assert t.init('proj', 'run', {}, str(tmp_path)) is True
    assert 'space_id' not in fake.kwargs and 'server_url' not in fake.kwargs
    assert (tmp_path / 'trackio').exists()


def test_trackio_space_id_is_forwarded(monkeypatch, tmp_path):
    fake = _FakeTrackio()
    monkeypatch.setitem(sys.modules, 'trackio', fake)
    t = tracking.TrackioTracker({'trackio_space_id': 'me/dash'})
    t.init('proj', 'run', {}, str(tmp_path))
    assert fake.kwargs['space_id'] == 'me/dash'


def test_trackio_server_url_is_forwarded(monkeypatch, tmp_path):
    fake = _FakeTrackio()
    monkeypatch.setitem(sys.modules, 'trackio', fake)
    t = tracking.TrackioTracker({'trackio_server_url': 'http://host:7860'})
    t.init('proj', 'run', {}, str(tmp_path))
    assert fake.kwargs['server_url'] == 'http://host:7860'


def test_trackio_init_failure_disables_without_raising(monkeypatch, tmp_path):
    fake = _FakeTrackio(fail=True)
    monkeypatch.setitem(sys.modules, 'trackio', fake)
    t = tracking.TrackioTracker({})
    assert t.init('proj', 'run', {}, str(tmp_path)) is False
    assert t.enabled is False


def test_trackio_missing_package_disables_without_raising(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == 'trackio':
            raise ImportError('no module named trackio')
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', fake_import)
    t = tracking.TrackioTracker({})
    assert t.init('proj', 'run', {}, str(tmp_path)) is False
    assert t.enabled is False