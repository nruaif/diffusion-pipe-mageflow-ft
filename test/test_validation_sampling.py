"""Offline tests for utils/validation_sampling.py.

Pure logic only: config parsing, the per-prompt override merge, the sigma
schedule, seed policy, and cadence gating. No GPU, no model, no files.
Run with: python3 -m pytest test/test_validation_sampling.py -q
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.validation_sampling import (  # noqa: E402
    DEFAULT_SAMPLE_SETTINGS,
    SamplingConfigError,
    build_sampling_config,
    build_sigmas,
    combine_cfg,
    euler_step,
    resolve_seed,
    should_sample,
    uses_negative_branch,
)


# --- disabled / absent -----------------------------------------------------

def test_no_sampling_table_returns_none():
    assert build_sampling_config({}) is None


def test_explicitly_disabled_returns_none():
    assert build_sampling_config({'sampling': {'enable': False, 'prompts': ['a']}}) is None


# --- the override merge (the core requirement) -----------------------------

def test_bare_string_prompt_inherits_all_defaults():
    cfg = build_sampling_config({'sampling': {'prompts': ['a cat']}})
    p = cfg['prompts'][0]
    assert p['prompt'] == 'a cat'
    for key, default in DEFAULT_SAMPLE_SETTINGS.items():
        assert p[key] == default, key


def test_run_level_overrides_defaults_for_every_prompt():
    cfg = build_sampling_config({'sampling': {
        'steps': 12, 'cfg': 3.0, 'width': 768, 'height': 512,
        'prompts': ['a cat', 'a dog'],
    }})
    for p in cfg['prompts']:
        assert (p['steps'], p['cfg'], p['width'], p['height']) == (12, 3.0, 768, 512)
        # untouched keys still fall back to the library default
        assert p['shift'] == DEFAULT_SAMPLE_SETTINGS['shift']


def test_per_prompt_overrides_beat_run_level():
    cfg = build_sampling_config({'sampling': {
        'steps': 12, 'cfg': 3.0, 'shift': 6.0,
        'prompts': [
            'inherits everything',
            {'prompt': 'custom steps', 'steps': 50},
            {'prompt': 'custom shift only', 'shift': 2.5},
            {'prompt': 'fully custom', 'steps': 4, 'cfg': 1.0, 'shift': 1.0,
             'width': 512, 'height': 512, 'seed': 7},
        ],
    }})
    a, b, c, d = cfg['prompts']

    # a: inherits run-level
    assert (a['steps'], a['cfg'], a['shift']) == (12, 3.0, 6.0)
    # b: overrides steps only, inherits cfg/shift
    assert (b['steps'], b['cfg'], b['shift']) == (50, 3.0, 6.0)
    # c: overrides shift only — the partial-override case that matters most
    assert (c['steps'], c['cfg'], c['shift']) == (12, 3.0, 2.5)
    # d: overrides everything
    assert (d['steps'], d['cfg'], d['shift'], d['width'], d['seed']) == (4, 1.0, 1.0, 512, 7)


def test_per_prompt_override_does_not_leak_between_prompts():
    cfg = build_sampling_config({'sampling': {'prompts': [
        {'prompt': 'first', 'steps': 99},
        {'prompt': 'second'},
    ]}})
    assert cfg['prompts'][0]['steps'] == 99
    assert cfg['prompts'][1]['steps'] == DEFAULT_SAMPLE_SETTINGS['steps']


def test_negative_prompt_is_overridable_per_prompt():
    cfg = build_sampling_config({'sampling': {
        'negative_prompt': 'blurry',
        'prompts': ['inherits', {'prompt': 'custom', 'negative_prompt': 'worst quality'}],
    }})
    assert cfg['prompts'][0]['negative_prompt'] == 'blurry'
    assert cfg['prompts'][1]['negative_prompt'] == 'worst quality'


# --- validation errors surface at startup ----------------------------------

def test_enabled_without_prompts_raises():
    with pytest.raises(SamplingConfigError, match='no prompts'):
        build_sampling_config({'sampling': {'enable': True}})


def test_unknown_run_level_key_raises():
    with pytest.raises(SamplingConfigError, match='unknown key'):
        build_sampling_config({'sampling': {'prompts': ['a'], 'stpes': 20}})


def test_unknown_per_prompt_key_raises():
    with pytest.raises(SamplingConfigError, match='unknown key'):
        build_sampling_config({'sampling': {'prompts': [{'prompt': 'a', 'cfgg': 3}]}})


def test_prompt_entry_without_prompt_key_raises():
    with pytest.raises(SamplingConfigError, match='no "prompt" key'):
        build_sampling_config({'sampling': {'prompts': [{'steps': 20}]}})


def test_non_multiple_of_16_resolution_raises_with_suggestion():
    with pytest.raises(SamplingConfigError, match='multiple of 16'):
        build_sampling_config({'sampling': {'prompts': ['a'], 'width': 1000}})


def test_bad_steps_raises():
    with pytest.raises(SamplingConfigError, match='steps'):
        build_sampling_config({'sampling': {'prompts': ['a'], 'steps': 0}})


def test_bad_seed_strategy_raises():
    with pytest.raises(SamplingConfigError, match='seed_strategy'):
        build_sampling_config({'sampling': {'prompts': ['a'], 'seed_strategy': 'random'}})


def test_no_cadence_at_all_raises():
    with pytest.raises(SamplingConfigError, match='no cadence'):
        build_sampling_config({'sampling': {
            'prompts': ['a'], 'sample_every_n_epochs': 0, 'sample_at_first': False}})


# --- labels ----------------------------------------------------------------

def test_label_defaults_to_sanitized_prompt():
    cfg = build_sampling_config({'sampling': {'prompts': ['a cat, sitting on a mat!']}})
    label = cfg['prompts'][0]['label']
    assert ' ' not in label and ',' not in label and '!' not in label
    assert label.startswith('a_cat')


def test_explicit_label_is_kept():
    cfg = build_sampling_config({'sampling': {'prompts': [{'prompt': 'x', 'label': 'my_label'}]}})
    assert cfg['prompts'][0]['label'] == 'my_label'


def test_label_survives_a_prompt_with_no_alphanumerics():
    cfg = build_sampling_config({'sampling': {'prompts': ['!!!', '???']}})
    assert cfg['prompts'][0]['label'] == 'prompt0'
    assert cfg['prompts'][1]['label'] == 'prompt1'


# --- cadence ---------------------------------------------------------------

def test_cadence_none_config_never_samples():
    assert should_sample(None, 10, 1, True) is False


def test_epoch_cadence_only_fires_on_finished_epoch():
    cfg = build_sampling_config({'sampling': {'prompts': ['a'], 'sample_every_n_epochs': 2}})
    assert should_sample(cfg, 100, 2, finished_epoch=True) is True
    assert should_sample(cfg, 100, 2, finished_epoch=False) is False
    assert should_sample(cfg, 100, 3, finished_epoch=True) is False


def test_step_cadence_fires_independently_of_epochs():
    cfg = build_sampling_config({'sampling': {
        'prompts': ['a'], 'sample_every_n_steps': 50, 'sample_every_n_epochs': 0}})
    assert should_sample(cfg, 50, 1, finished_epoch=False) is True
    assert should_sample(cfg, 51, 1, finished_epoch=False) is False


# --- seed policy -----------------------------------------------------------

def test_fixed_seed_is_stable_across_rounds():
    cfg = build_sampling_config({'sampling': {'prompts': [{'prompt': 'a', 'seed': 123}]}})
    p = cfg['prompts'][0]
    assert [resolve_seed(p, cfg, r) for r in range(4)] == [123] * 4


def test_walk_seed_advances_per_round():
    cfg = build_sampling_config({'sampling': {
        'prompts': [{'prompt': 'a', 'seed': 100}], 'seed_strategy': 'walk'}})
    p = cfg['prompts'][0]
    assert [resolve_seed(p, cfg, r) for r in range(4)] == [100, 101, 102, 103]


def test_different_prompts_can_use_different_seeds():
    cfg = build_sampling_config({'sampling': {'prompts': [
        {'prompt': 'a', 'seed': 1}, {'prompt': 'b', 'seed': 2}, 'c']}})
    seeds = [resolve_seed(p, cfg, 0) for p in cfg['prompts']]
    assert seeds == [1, 2, DEFAULT_SAMPLE_SETTINGS['seed']]


# --- schedule / integration ------------------------------------------------

def test_sigmas_shape_and_endpoints():
    s = build_sigmas(20, 6.0)
    assert s.shape == (21,)
    assert s[0].item() == pytest.approx(1.0)
    assert s[-1].item() == 0.0


def test_sigmas_strictly_descending():
    for shift in (1.0, 3.0, 6.0, 12.0):
        s = build_sigmas(30, shift)
        assert torch.all(s[1:] < s[:-1]), f'not descending at shift={shift}'


def test_shift_of_one_is_linear():
    s = build_sigmas(4, 1.0)
    assert s.tolist() == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])


def test_higher_shift_spends_more_steps_at_high_noise():
    low, high = build_sigmas(10, 1.0), build_sigmas(10, 6.0)
    # every interior sigma sits higher under the larger shift
    assert torch.all(high[1:-1] > low[1:-1])


def test_single_step_schedule_is_valid():
    s = build_sigmas(1, 6.0)
    assert s.shape == (2,) and s[0].item() == pytest.approx(1.0) and s[-1].item() == 0.0


def test_euler_step_reaches_target_in_one_step_for_constant_velocity():
    # With v constant, integrating 1 -> 0 should move the latent by exactly -v.
    x = torch.zeros(1, 4, 8)
    v = torch.ones(1, 4, 8)
    out = euler_step(x, v, torch.tensor(1.0), torch.tensor(0.0))
    assert torch.allclose(out, -v)


# --- CFG -------------------------------------------------------------------

def test_cfg_of_one_returns_conditional():
    cond, unc = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    assert torch.allclose(combine_cfg(cond, unc, 1.0), cond, atol=1e-6)


def test_cfg_of_zero_returns_unconditional():
    cond, unc = torch.randn(1, 4, 8), torch.randn(1, 4, 8)
    assert torch.allclose(combine_cfg(cond, unc, 0.0), unc, atol=1e-6)


def test_cfg_extrapolates_beyond_conditional():
    cond, unc = torch.ones(1, 1, 4), torch.zeros(1, 1, 4)
    assert torch.allclose(combine_cfg(cond, unc, 3.0), torch.full((1, 1, 4), 3.0))


def test_renormalization_restores_conditional_norm():
    torch.manual_seed(0)
    cond, unc = torch.randn(1, 6, 16), torch.randn(1, 6, 16)
    out = combine_cfg(cond, unc, 7.0, renormalize=True)
    assert torch.allclose(
        torch.norm(out, dim=-1), torch.norm(cond, dim=-1), rtol=1e-4)


def test_negative_branch_only_used_above_cfg_one():
    assert uses_negative_branch(5.0, '') is True
    assert uses_negative_branch(1.0, 'blurry') is False
    assert uses_negative_branch(0.5, 'blurry') is False
