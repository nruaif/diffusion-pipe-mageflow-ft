"""End-to-end tests for the validation sampling driver, using a mock adapter.

No GPU, no model weights, no real VAE. These cover the parts of the feature
that are easy to get subtly wrong and expensive to discover on a real training
run: seed determinism, CFG forward-pass counts, per-prompt overrides actually
changing work done, block-swap enter/exit ordering, RNG isolation from the
training noise stream, and per-prompt failure containment.

Run with: python3 -m pytest test/test_sampling_driver.py -q
"""

import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.validation_sampling import (  # noqa: E402
    SamplingAdapter,
    build_sampling_config,
    generate_and_log_samples,
    sample_one,
)

CPU = torch.device('cpu')


class MockAdapter(SamplingAdapter):
    """Deterministic stand-in for a real model.

    The velocity is a fixed multiple of the current latents, so the trajectory
    depends on the initial noise — which means the decoded output is a direct
    function of the seed, and seed handling is actually observable.
    """

    def __init__(self, fail_on=None):
        self.velocity_calls = 0
        self.init_noise = []
        self.fail_on = fail_on

    def encode_prompt(self, prompt):
        return torch.tensor(float(len(prompt) % 7) + 1.0)

    def init_latents(self, width, height, generator, device, dtype):
        h, w = height // 16, width // 16
        x = torch.randn(1, h * w, 8, generator=generator, dtype=torch.float32)
        self.init_noise.append(x.clone())
        return x

    def predict_velocity(self, latents, text_cond, sigma, width, height):
        if self.fail_on is not None and float(text_cond) == self.fail_on:
            raise RuntimeError('simulated model failure')
        self.velocity_calls += 1
        return latents * 0.1 * float(text_cond)

    def decode(self, latents, width, height):
        v = latents.mean().item()
        c = max(0, min(255, int((v + 3) / 6 * 255)))
        return Image.new('RGB', (width, height), (c, c, c))


class MockModel:
    def __init__(self, adapter=None):
        self.adapter = adapter or MockAdapter()
        self.swap_calls = []

    def get_sampling_adapter(self):
        return self.adapter

    def sampling_device(self):
        return CPU

    def sampling_dtype(self):
        return torch.float32

    def prepare_block_swap_inference(self, disable_block_swap=False):
        self.swap_calls.append('inference')

    def prepare_block_swap_training(self):
        self.swap_calls.append('training')


@pytest.fixture
def cfg():
    return build_sampling_config({'sampling': {
        'steps': 6, 'cfg': 4.0, 'width': 256, 'height': 256,
        'prompts': [
            'a cat on a mat',
            {'prompt': 'a dog', 'steps': 3, 'seed': 999},
            {'prompt': 'no guidance', 'cfg': 1.0},
        ],
    }})


# --- seeding ---------------------------------------------------------------

def test_same_seed_produces_identical_init_noise(cfg):
    a = MockAdapter()
    for seed in (42, 42):
        sample_one(a, cfg['prompts'][0], seed, CPU, torch.float32)
    assert torch.equal(a.init_noise[0], a.init_noise[1])


def test_different_seed_produces_different_init_noise(cfg):
    a = MockAdapter()
    sample_one(a, cfg['prompts'][0], 42, CPU, torch.float32)
    sample_one(a, cfg['prompts'][0], 43, CPU, torch.float32)
    assert not torch.equal(a.init_noise[0], a.init_noise[1])


def test_same_seed_produces_identical_output_image(cfg):
    a = MockAdapter()
    i1 = sample_one(a, cfg['prompts'][0], 42, CPU, torch.float32)
    i2 = sample_one(a, cfg['prompts'][0], 42, CPU, torch.float32)
    assert list(i1.convert('RGB').tobytes()) == list(i2.convert('RGB').tobytes())


def test_init_noise_is_unit_normal(cfg):
    a = MockAdapter()
    sample_one(a, cfg['prompts'][0], 0, CPU, torch.float32)
    assert a.init_noise[0].std().item() == pytest.approx(1.0, abs=0.1)


# --- CFG cost --------------------------------------------------------------

def test_cfg_above_one_doubles_forward_passes(cfg):
    a = MockAdapter()
    sample_one(a, cfg['prompts'][0], 1, CPU, torch.float32)  # steps=6, cfg=4
    assert a.velocity_calls == 12


def test_cfg_of_one_skips_the_negative_pass(cfg):
    a = MockAdapter()
    sample_one(a, cfg['prompts'][2], 1, CPU, torch.float32)  # steps=6, cfg=1
    assert a.velocity_calls == 6


def test_per_prompt_steps_override_changes_work_done(cfg):
    a = MockAdapter()
    sample_one(a, cfg['prompts'][1], 1, CPU, torch.float32)  # steps=3, cfg=4
    assert a.velocity_calls == 6


# --- output geometry -------------------------------------------------------

def test_output_image_matches_requested_resolution():
    c = build_sampling_config({'sampling': {
        'prompts': [{'prompt': 'x', 'width': 512, 'height': 768}], 'steps': 2}})
    img = sample_one(MockAdapter(), c['prompts'][0], 1, CPU, torch.float32)
    assert img.size == (512, 768)


# --- driver ----------------------------------------------------------------

def test_driver_writes_one_png_per_prompt(cfg, tmp_path):
    generate_and_log_samples(MockModel(), cfg, None, 100, tmp_path, 'epoch1', 0)
    files = sorted(p.name for p in (tmp_path / 'samples' / 'epoch1').glob('*.png'))
    assert len(files) == 3
    assert files[0].startswith('00_') and files[1].startswith('01_')
    assert 'seed999' in files[1]  # per-prompt seed override lands in the filename


def test_driver_restores_training_mode_after_sampling(cfg, tmp_path):
    m = MockModel()
    generate_and_log_samples(m, cfg, None, 1, tmp_path, 'e', 0)
    assert m.swap_calls == ['inference', 'training']


def test_driver_restores_training_mode_even_when_every_prompt_fails(cfg, tmp_path):
    # Block swap must be restored or the next training step runs in the wrong mode.
    class AlwaysFails(MockAdapter):
        def predict_velocity(self, *a, **k):
            raise RuntimeError('boom')
    m = MockModel(AlwaysFails())
    generate_and_log_samples(m, cfg, None, 1, tmp_path, 'e', 0)
    assert m.swap_calls == ['inference', 'training']


def test_one_failing_prompt_does_not_abort_the_round(cfg, tmp_path):
    failing = float(len('a dog') % 7) + 1.0
    m = MockModel(MockAdapter(fail_on=failing))
    generate_and_log_samples(m, cfg, None, 1, tmp_path, 'e', 0)
    assert len(list((tmp_path / 'samples' / 'e').glob('*.png'))) == 2


def test_sampling_does_not_perturb_the_training_rng_stream(cfg, tmp_path):
    torch.manual_seed(7)
    expected = torch.randn(3)

    torch.manual_seed(7)
    generate_and_log_samples(MockModel(), cfg, None, 1, tmp_path, 'e', 0)
    actual = torch.randn(3)

    assert torch.equal(expected, actual)


def test_unsupported_model_is_skipped_silently(cfg, tmp_path):
    class Unsupported(MockModel):
        def get_sampling_adapter(self):
            return None
    m = Unsupported()
    generate_and_log_samples(m, cfg, None, 1, tmp_path, 'e', 0)
    assert m.swap_calls == []
    assert not (tmp_path / 'samples').exists()


def test_disk_logging_can_be_disabled(tmp_path):
    c = build_sampling_config({'sampling': {
        'prompts': ['x'], 'steps': 2, 'save_to_disk': False}})
    generate_and_log_samples(MockModel(), c, None, 1, tmp_path, 'e', 0)
    assert not (tmp_path / 'samples').exists()


def test_wandb_failure_does_not_break_training(cfg, tmp_path):
    class ExplodingWandb:
        enabled = True

        @staticmethod
        def Image(*a, **k):
            raise RuntimeError('tracker is down')

        @staticmethod
        def log(*a, **k):
            raise RuntimeError('tracker is down')

    # Must not raise, and must still write the disk archive.
    generate_and_log_samples(MockModel(), cfg, None, 1, tmp_path, 'e', 0,
                             tracker=ExplodingWandb())
    assert len(list((tmp_path / 'samples' / 'e').glob('*.png'))) == 3


def test_walk_seed_changes_output_between_rounds(tmp_path):
    c = build_sampling_config({'sampling': {
        'prompts': [{'prompt': 'x', 'seed': 5}], 'steps': 2, 'cfg': 1.0,
        'seed_strategy': 'walk'}})
    a = MockAdapter()
    m = MockModel(a)
    generate_and_log_samples(m, c, None, 1, tmp_path, 'r0', round_index=0)
    generate_and_log_samples(m, c, None, 2, tmp_path, 'r1', round_index=1)
    assert not torch.equal(a.init_noise[0], a.init_noise[1])


def test_fixed_seed_reproduces_noise_between_rounds(tmp_path):
    c = build_sampling_config({'sampling': {
        'prompts': [{'prompt': 'x', 'seed': 5}], 'steps': 2, 'cfg': 1.0}})
    a = MockAdapter()
    m = MockModel(a)
    generate_and_log_samples(m, c, None, 1, tmp_path, 'r0', round_index=0)
    generate_and_log_samples(m, c, None, 2, tmp_path, 'r1', round_index=1)
    assert torch.equal(a.init_noise[0], a.init_noise[1])


# --- wandb call shape (regression guards for the step-alignment fix) --------

class RecordingWandb:
    """Mock tracker that enforces the two properties the step fix depends on."""

    enabled = True

    def __init__(self):
        self.calls = []

    def log(self, data, step=None):
        assert step is not None, 'wandb.log() called without step='
        assert 'step' not in data, "'step' passed as a dict key instead of step="
        self.calls.append((step, sorted(data.keys())))

    def Image(self, *a, **k):
        return ('Image', a, k)


def test_sampling_round_makes_exactly_one_wandb_call(cfg, tmp_path):
    # Previously this cost two calls (gallery, then sample_time_sec), which
    # advanced wandb's internal step counter twice for one logical step.
    wb = RecordingWandb()
    generate_and_log_samples(MockModel(), cfg, None, 42, tmp_path, 'r', 0, tracker=wb)
    assert len(wb.calls) == 1, wb.calls
    step, keys = wb.calls[0]
    assert step == 42
    assert keys == ['samples', 'samples/sample_time_sec']


def test_all_prompts_failing_still_logs_timing_at_the_right_step(cfg, tmp_path):
    # The round must stay visible on wandb's x-axis even with zero images,
    # and must not create an empty samples/ directory on disk.
    class AlwaysFails(MockAdapter):
        def predict_velocity(self, *a, **k):
            raise RuntimeError('boom')

    wb = RecordingWandb()
    generate_and_log_samples(MockModel(AlwaysFails()), cfg, None, 13, tmp_path, 'r', 0,
                             tracker=wb)
    assert len(wb.calls) == 1, wb.calls
    step, keys = wb.calls[0]
    assert step == 13
    assert keys == ['samples/sample_time_sec']
    assert not (tmp_path / 'samples').exists()


def test_log_to_wandb_false_makes_no_wandb_calls(tmp_path):
    c = build_sampling_config({'sampling': {
        'prompts': ['a'], 'steps': 2, 'cfg': 1.0, 'log_to_wandb': False}})
    wb = RecordingWandb()
    generate_and_log_samples(MockModel(), c, None, 5, tmp_path, 'r', 0, tracker=wb)
    assert wb.calls == []